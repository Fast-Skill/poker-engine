//! Equity: how often each holding wins the pot, given everything already known.
//!
//! Four entry points, in increasing order of what they don't know:
//!
//! * [`exact`] — specific hands, every remaining board enumerated. Ground truth.
//! * [`monte_carlo`] — specific hands, boards sampled. For when enumeration is
//!   too wide to sit through.
//! * [`range_vs_range_exact`] — ranges instead of hands, enumerated. Only
//!   tractable from the turn on.
//! * [`range_vs_range`] — ranges, sampled. The general case.
//!
//! Every entry point takes explicit `dead` cards, because card removal is the
//! whole difficulty of equity work. An opponent holding two aces changes how
//! often you make a set; a mucked hand you happened to see removes those cards
//! from every runout. Nothing here ever deals a card that is already known, and
//! the sampling deliberately preserves the *joint* distribution over holdings
//! rather than each player's marginal — see [`range_vs_range`] for why that
//! distinction is not pedantic.
//!
//! Equity is reported as a fraction of the pot, so a k-way tie pays each of the
//! winners `1/k`. That differs from [`crate::game::Game::payouts`], which
//! splits real chips and hands odd ones to the first seat left of the button.
//! Both are right: one is a long-run average, the other is a chip count.

use crate::card;
use crate::deck::Rng;
use crate::eval::eval;

/// Win, tie, and pot-share accounting for one equity calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct Equity {
    /// Boards enumerated, or trials sampled.
    pub trials: u64,
    /// Pot share accumulated per player: `1` for an outright win, `1/k` when
    /// tied k ways. Divide by `trials` — or just call [`Equity::share`].
    pub equity: Vec<f64>,
    /// Trials this player won outright.
    pub wins: Vec<u64>,
    /// Trials this player was among several tied for best.
    pub ties: Vec<u64>,
}

impl Equity {
    fn new(players: usize) -> Equity {
        Equity {
            trials: 0,
            equity: vec![0.0; players],
            wins: vec![0; players],
            ties: vec![0; players],
        }
    }

    pub fn players(&self) -> usize {
        self.equity.len()
    }

    /// Fraction of the pot player `i` takes on average, in `0.0..=1.0`.
    pub fn share(&self, i: usize) -> f64 {
        if self.trials == 0 {
            0.0
        } else {
            self.equity[i] / self.trials as f64
        }
    }

    pub fn shares(&self) -> Vec<f64> {
        (0..self.players()).map(|i| self.share(i)).collect()
    }

    fn record(&mut self, strengths: &[u32]) {
        let best = strengths.iter().copied().max().expect("no players");
        let n_best = strengths.iter().filter(|&&s| s == best).count();
        if n_best == 1 {
            let i = strengths.iter().position(|&s| s == best).expect("best exists");
            self.wins[i] += 1;
            self.equity[i] += 1.0;
        } else {
            let split = 1.0 / n_best as f64;
            for (i, &s) in strengths.iter().enumerate() {
                if s == best {
                    self.ties[i] += 1;
                    self.equity[i] += split;
                }
            }
        }
        self.trials += 1;
    }
}

/// Mark `cards` as known, rejecting anything already marked. Duplicated cards
/// are the single most common way to get a silently wrong equity number, so
/// they are a hard error rather than a filtered-out case.
fn block(seen: &mut [bool; 52], cards: &[u8], what: &str) {
    for &c in cards {
        assert!(c < 52, "{what}: {c} is not a card");
        assert!(!seen[c as usize], "{what}: {} appears twice", card::to_string(c));
        seen[c as usize] = true;
    }
}

fn unseen(seen: &[bool; 52]) -> Vec<u8> {
    (0..52u8).filter(|&c| !seen[c as usize]).collect()
}

/// Shared validation for the specific-hands entry points. Returns the cards
/// still in the deck, a board buffer with the known cards already in place, and
/// how many more are needed.
fn setup(hands: &[[u8; 2]], board: &[u8], dead: &[u8]) -> ([bool; 52], [u8; 5], usize) {
    assert!(hands.len() >= 2, "equity needs at least two hands");
    assert!(board.len() <= 5, "a board holds at most five cards");

    let mut seen = [false; 52];
    for (i, h) in hands.iter().enumerate() {
        block(&mut seen, h, &format!("hand {i}"));
    }
    block(&mut seen, board, "board");
    block(&mut seen, dead, "dead cards");

    let mut full = [card::NO_CARD; 5];
    full[..board.len()].copy_from_slice(board);
    (seen, full, 5 - board.len())
}

#[inline]
fn strengths_of(hands: &[[u8; 2]], full: &[u8; 5], out: &mut Vec<u32>) {
    out.clear();
    for h in hands {
        out.push(eval(&[full[0], full[1], full[2], full[3], full[4], h[0], h[1]]));
    }
}

/// Step `idx` to the next combination of its length drawn from `0..n`, in
/// lexicographic order. Returns false once the last one has been visited, so
/// the caller evaluates first and advances after.
fn next_combination(idx: &mut [usize], n: usize) -> bool {
    let k = idx.len();
    if k == 0 {
        return false;
    }
    let mut i = k - 1;
    loop {
        // Position i can climb to `n - k + i` and still leave room for the
        // positions after it.
        if idx[i] < n - k + i {
            idx[i] += 1;
            for j in i + 1..k {
                idx[j] = idx[j - 1] + 1;
            }
            return true;
        }
        if i == 0 {
            return false;
        }
        i -= 1;
    }
}

/// Exact equity for specific hands: every remaining board, weighted equally.
///
/// Cost is `C(remaining, 5 - board.len())`. Heads-up preflop that is
/// `C(48, 5)` = 1,712,304 boards, which runs in well under a second; on the
/// flop it is 990 and on the turn 44.
pub fn exact(hands: &[[u8; 2]], board: &[u8], dead: &[u8]) -> Equity {
    let (seen, mut full, k) = setup(hands, board, dead);
    let avail = unseen(&seen);
    assert!(avail.len() >= k, "not enough cards left to finish the board");

    let base = board.len();
    let mut eq = Equity::new(hands.len());
    let mut strengths = Vec::with_capacity(hands.len());
    let mut idx: Vec<usize> = (0..k).collect();

    loop {
        for (j, &ix) in idx.iter().enumerate() {
            full[base + j] = avail[ix];
        }
        strengths_of(hands, &full, &mut strengths);
        eq.record(&strengths);
        if !next_combination(&mut idx, avail.len()) {
            break;
        }
    }
    eq
}

/// Sampled equity for specific hands. Deterministic given `seed`.
pub fn monte_carlo(
    hands: &[[u8; 2]],
    board: &[u8],
    dead: &[u8],
    trials: u64,
    seed: u64,
) -> Equity {
    let (seen, mut full, k) = setup(hands, board, dead);
    let mut avail = unseen(&seen);
    assert!(avail.len() >= k, "not enough cards left to finish the board");

    let base = board.len();
    let mut rng = Rng::new(seed);
    let mut eq = Equity::new(hands.len());
    let mut strengths = Vec::with_capacity(hands.len());

    for _ in 0..trials {
        // Partial Fisher-Yates: only the k cards actually needed get shuffled
        // into place. `avail` stays permuted between trials, which costs
        // nothing — any prefix of a uniform permutation is a uniform sample.
        for j in 0..k {
            let pick = j + rng.below((avail.len() - j) as u64) as usize;
            avail.swap(j, pick);
            full[base + j] = avail[j];
        }
        strengths_of(hands, &full, &mut strengths);
        eq.record(&strengths);
    }
    eq
}

/// A set of two-card holdings. Combos are stored canonically as
/// `[higher card, lower card]`, sorted and deduplicated, so two ranges written
/// differently but meaning the same thing compare equal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Range {
    combos: Vec<[u8; 2]>,
}

impl Range {
    pub fn combos(&self) -> &[[u8; 2]] {
        &self.combos
    }

    pub fn len(&self) -> usize {
        self.combos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.combos.is_empty()
    }

    /// All 1,326 two-card combinations.
    pub fn all() -> Range {
        let mut combos = Vec::with_capacity(1326);
        for a in 0..52u8 {
            for b in 0..a {
                combos.push([a, b]);
            }
        }
        Range { combos }
    }

    /// Drop every combo using one of `blocked`. This is card removal at the
    /// range level: a king on the board is a king no opponent can hold.
    pub fn without(&self, blocked: &[u8]) -> Range {
        let mut gone = [false; 52];
        for &c in blocked {
            if c < 52 {
                gone[c as usize] = true;
            }
        }
        Range {
            combos: self
                .combos
                .iter()
                .copied()
                .filter(|c| !gone[c[0] as usize] && !gone[c[1] as usize])
                .collect(),
        }
    }

    /// Parse standard range notation. Tokens are separated by commas or
    /// whitespace and may be:
    ///
    /// | Form | Example | Meaning |
    /// | --- | --- | --- |
    /// | pair | `AA` | all six combos |
    /// | suited / offsuit | `AKs`, `AKo` | four / twelve combos |
    /// | unqualified | `AK` | all sixteen |
    /// | explicit combo | `AsKh` | exactly that one |
    /// | pair and up | `TT+` | `TT` through `AA` |
    /// | kicker and up | `AJs+` | `AJs`, `AQs`, `AKs` |
    /// | span | `77-JJ`, `AJs-A9s` | inclusive both ends |
    ///
    /// Spans of non-pairs must share their high card. Gapped-connector spans
    /// such as `T9s-76s` are not accepted, because the notation is ambiguous
    /// about whether the gap or the ranks are what is being held fixed.
    pub fn parse(spec: &str) -> Result<Range, String> {
        let mut combos = Vec::new();
        for tok in spec
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|t| !t.is_empty())
        {
            parse_token(tok, &mut combos)?;
        }
        combos.sort_unstable();
        combos.dedup();
        Ok(Range { combos })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Suits {
    Suited,
    Offsuit,
    Both,
}

fn rank_index(b: u8) -> Option<u8> {
    card::RANK_CHARS
        .iter()
        .position(|&r| r == b.to_ascii_uppercase())
        .map(|p| p as u8)
}

fn emit(hi: u8, lo: u8, suits: Suits, out: &mut Vec<[u8; 2]>) {
    if hi == lo {
        for s1 in 0..4u8 {
            for s2 in 0..s1 {
                out.push([card::card(hi, s1), card::card(hi, s2)]);
            }
        }
        return;
    }
    match suits {
        Suits::Suited => {
            for s in 0..4u8 {
                out.push([card::card(hi, s), card::card(lo, s)]);
            }
        }
        Suits::Offsuit => {
            for a in 0..4u8 {
                for b in 0..4u8 {
                    if a != b {
                        out.push([card::card(hi, a), card::card(lo, b)]);
                    }
                }
            }
        }
        Suits::Both => {
            emit(hi, lo, Suits::Suited, out);
            emit(hi, lo, Suits::Offsuit, out);
        }
    }
}

/// Parse a hand class such as `AA`, `AKs`, or `T9o` into `(high, low, suits)`.
fn shorthand(tok: &str) -> Result<(u8, u8, Suits), String> {
    let b = tok.as_bytes();
    if b.len() < 2 || b.len() > 3 {
        return Err(format!("`{tok}` is not a hand class like `AA`, `AKs`, or `T9o`"));
    }
    let r1 = rank_index(b[0]).ok_or_else(|| format!("`{tok}`: `{}` is not a rank", b[0] as char))?;
    let r2 = rank_index(b[1]).ok_or_else(|| format!("`{tok}`: `{}` is not a rank", b[1] as char))?;
    let suits = if b.len() == 3 {
        match b[2].to_ascii_lowercase() {
            b's' => Suits::Suited,
            b'o' => Suits::Offsuit,
            other => {
                return Err(format!("`{tok}` ends in `{}`, expected `s` or `o`", other as char))
            }
        }
    } else {
        Suits::Both
    };
    if r1 == r2 && suits != Suits::Both {
        return Err(format!("`{tok}`: a pair cannot be suited or offsuit"));
    }
    Ok(if r1 >= r2 { (r1, r2, suits) } else { (r2, r1, suits) })
}

fn parse_token(tok: &str, out: &mut Vec<[u8; 2]>) -> Result<(), String> {
    if !tok.is_ascii() {
        return Err(format!("`{tok}` is not ASCII"));
    }

    // An explicit combo like `AsKh`. Tried first, but note `AJs+` is also four
    // bytes — there `J` is not a suit, the card parse fails, and we fall
    // through to the shorthand forms below.
    if tok.len() == 4 {
        if let (Some(a), Some(b)) = (card::parse(&tok[0..2]), card::parse(&tok[2..4])) {
            if a == b {
                return Err(format!("`{tok}` uses the same card twice"));
            }
            out.push(if a > b { [a, b] } else { [b, a] });
            return Ok(());
        }
    }

    if let Some(base) = tok.strip_suffix('+') {
        let (hi, lo, suits) = shorthand(base)?;
        if hi == lo {
            for r in hi..13 {
                emit(r, r, suits, out);
            }
        } else {
            // The high card is pinned and the kicker climbs to just under it.
            for l in lo..hi {
                emit(hi, l, suits, out);
            }
        }
        return Ok(());
    }

    if let Some((a, b)) = tok.split_once('-') {
        let (hi1, lo1, s1) = shorthand(a)?;
        let (hi2, lo2, s2) = shorthand(b)?;
        if s1 != s2 {
            return Err(format!("`{tok}` mixes suited and offsuit ends"));
        }
        if (hi1 == lo1) != (hi2 == lo2) {
            return Err(format!("`{tok}` mixes pairs and non-pairs"));
        }
        let (from, to) = if lo1 <= lo2 { (lo1, lo2) } else { (lo2, lo1) };
        if hi1 == lo1 {
            for r in from..=to {
                emit(r, r, s1, out);
            }
        } else {
            if hi1 != hi2 {
                return Err(format!("`{tok}`: both ends must share a high card, like `AJs-A9s`"));
            }
            for l in from..=to {
                emit(hi1, l, s1, out);
            }
        }
        return Ok(());
    }

    let (hi, lo, suits) = shorthand(tok)?;
    emit(hi, lo, suits, out);
    Ok(())
}

/// Give up on a trial after this many conflicting draws. Only reachable when
/// the ranges genuinely cannot be dealt together, e.g. both sides holding
/// exactly `AsAh`.
const MAX_DRAW_ATTEMPTS: u32 = 10_000;

fn prepare_ranges(ranges: &[Range], board: &[u8], dead: &[u8]) -> ([bool; 52], Vec<Range>) {
    assert!(ranges.len() >= 2, "equity needs at least two ranges");
    assert!(board.len() <= 5, "a board holds at most five cards");

    let mut seen = [false; 52];
    block(&mut seen, board, "board");
    block(&mut seen, dead, "dead cards");

    // Removing known cards up front is not only an optimisation: it is what
    // keeps the rejection loop below from spinning on combos that could never
    // have been dealt.
    let blocked: Vec<u8> = (0..52u8).filter(|&c| seen[c as usize]).collect();
    let trimmed: Vec<Range> = ranges.iter().map(|r| r.without(&blocked)).collect();
    for (i, r) in trimmed.iter().enumerate() {
        assert!(
            !r.is_empty(),
            "range {i} has no combos left once the board and dead cards are removed"
        );
    }
    (seen, trimmed)
}

/// Sampled equity for ranges rather than specific hands.
///
/// Each trial draws one combo per range and then runs the board out. When two
/// players want the same card the **whole draw restarts**, and that detail is
/// the entire correctness argument.
///
/// Keeping the combos already drawn and resampling only the player that clashed
/// would bias the answer. It would let a holding that blocks most of the next
/// range appear just as often as one that blocks none of it, when in a real
/// deal the blocking holding is precisely the one that should show up less.
/// Restarting keeps the sample uniform over valid *joint* holdings, which is
/// what card removal actually means — and it is why this agrees with
/// [`range_vs_range_exact`] rather than being merely close to it.
pub fn range_vs_range(
    ranges: &[Range],
    board: &[u8],
    dead: &[u8],
    trials: u64,
    seed: u64,
) -> Equity {
    let (seen, ranges) = prepare_ranges(ranges, board, dead);
    let base = board.len();
    let k = 5 - base;
    let mut full = [card::NO_CARD; 5];
    full[..base].copy_from_slice(board);

    let mut rng = Rng::new(seed);
    let mut eq = Equity::new(ranges.len());
    let mut strengths = Vec::with_capacity(ranges.len());
    let mut hands = vec![[card::NO_CARD; 2]; ranges.len()];
    let mut avail: Vec<u8> = Vec::with_capacity(52);
    let mut used = [false; 52];

    for _ in 0..trials {
        let mut attempts = 0u32;
        'draw: loop {
            attempts += 1;
            assert!(
                attempts <= MAX_DRAW_ATTEMPTS,
                "these ranges cannot be dealt together — every draw collided"
            );
            used.copy_from_slice(&seen);
            for (i, r) in ranges.iter().enumerate() {
                let c = r.combos[rng.below(r.len() as u64) as usize];
                if used[c[0] as usize] || used[c[1] as usize] {
                    continue 'draw;
                }
                used[c[0] as usize] = true;
                used[c[1] as usize] = true;
                hands[i] = c;
            }
            break;
        }

        avail.clear();
        avail.extend((0..52u8).filter(|&c| !used[c as usize]));
        for j in 0..k {
            let pick = j + rng.below((avail.len() - j) as u64) as usize;
            avail.swap(j, pick);
            full[base + j] = avail[j];
        }
        strengths_of(&hands, &full, &mut strengths);
        eq.record(&strengths);
    }
    eq
}

/// Exact equity between two ranges: every valid pair of holdings against every
/// remaining board, all weighted equally.
///
/// Restricted to the turn and river on purpose. Two full ranges are 1,326
/// combos each, so preflop this would be roughly 1.7 million holding pairs
/// times 1.7 million boards — not a long wait, an impossible one. Use
/// [`range_vs_range`] before the turn.
pub fn range_vs_range_exact(ranges: &[Range], board: &[u8], dead: &[u8]) -> Equity {
    assert_eq!(ranges.len(), 2, "exact range equity is implemented for two ranges");
    assert!(
        (4..=5).contains(&board.len()),
        "exact range equity is only tractable from the turn on — use `range_vs_range` earlier"
    );

    let (seen, ranges) = prepare_ranges(ranges, board, dead);
    let base = board.len();
    let k = 5 - base;
    let mut full = [card::NO_CARD; 5];
    full[..base].copy_from_slice(board);

    let mut eq = Equity::new(2);
    let mut strengths = Vec::with_capacity(2);
    let mut avail: Vec<u8> = Vec::with_capacity(52);
    let mut idx: Vec<usize> = Vec::with_capacity(k);

    for &a in ranges[0].combos() {
        for &b in ranges[1].combos() {
            if a[0] == b[0] || a[0] == b[1] || a[1] == b[0] || a[1] == b[1] {
                continue;
            }
            let mut used = seen;
            for &c in a.iter().chain(b.iter()) {
                used[c as usize] = true;
            }
            avail.clear();
            avail.extend((0..52u8).filter(|&c| !used[c as usize]));

            idx.clear();
            idx.extend(0..k);
            let hands = [a, b];
            loop {
                for (j, &ix) in idx.iter().enumerate() {
                    full[base + j] = avail[ix];
                }
                strengths_of(&hands, &full, &mut strengths);
                eq.record(&strengths);
                if !next_combination(&mut idx, avail.len()) {
                    break;
                }
            }
        }
    }
    eq
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{parse, parse_many};

    fn hand(s: &str) -> [u8; 2] {
        let c = parse_many(s).unwrap();
        assert_eq!(c.len(), 2, "`{s}` is not two cards");
        [c[0], c[1]]
    }

    fn board(s: &str) -> Vec<u8> {
        parse_many(s).unwrap()
    }

    fn close(a: f64, b: f64, tol: f64, what: &str) {
        assert!((a - b).abs() <= tol, "{what}: {a:.5} vs {b:.5}, tolerance {tol}");
    }

    #[test]
    fn a_full_board_leaves_nothing_to_chance() {
        // Royal flush against trip aces, everything already dealt.
        let eq = exact(
            &[hand("JsTs"), hand("AcAd")],
            &board("AsKsQs2h3d"),
            &[],
        );
        assert_eq!(eq.trials, 1);
        assert_eq!(eq.share(0), 1.0);
        assert_eq!(eq.share(1), 0.0);
        assert_eq!(eq.wins, vec![1, 0]);
    }

    #[test]
    fn playing_the_board_splits() {
        // The board is a royal; neither hole card matters.
        let eq = exact(&[hand("2c3c"), hand("4d5d")], &board("AsKsQsJsTs"), &[]);
        assert_eq!(eq.trials, 1);
        assert_eq!(eq.shares(), vec![0.5, 0.5]);
        assert_eq!(eq.ties, vec![1, 1]);
        assert_eq!(eq.wins, vec![0, 0]);
    }

    #[test]
    fn drawing_dead_is_exactly_zero() {
        // Seat 0 already holds the royal on the turn, so the river is noise.
        // 52 - 4 board - 4 hole = 44 runouts.
        let eq = exact(&[hand("Ts9d"), hand("2c3c")], &board("AsKsQsJs"), &[]);
        assert_eq!(eq.trials, 44);
        assert_eq!(eq.share(0), 1.0);
        assert_eq!(eq.share(1), 0.0);
    }

    #[test]
    fn shares_always_sum_to_one() {
        let eq = exact(
            &[hand("AsAh"), hand("KsKd"), hand("7c2d")],
            &board("2s7h9c"),
            &[],
        );
        let total: f64 = eq.shares().iter().sum();
        close(total, 1.0, 1e-9, "three-way shares");
    }

    #[test]
    fn aces_over_kings_matches_the_published_equities() {
        // Checked against the standard equity tables rather than against our
        // own output, which is the only way a test like this says anything.
        // AA over KK is quoted as 81.25% to 82.64% depending on suits, and all
        // three configurations are pinned here because the spread is the part
        // that exercises suit handling.
        //
        // The direction is worth stating, because the intuition runs backwards:
        // sharing a suit with the kings *helps* the aces. In a shared suit the
        // aces hold the nut flush card, so both players draw to a flush only
        // one of them can win. Disjoint suits give the kings two clean flush
        // suits of their own, and that is the aces' worst case.
        let both = exact(&[hand("AsAh"), hand("KsKh")], &[], &[]);
        let one = exact(&[hand("AsAh"), hand("KsKd")], &[], &[]);
        let none = exact(&[hand("AsAh"), hand("KcKd")], &[], &[]);

        assert_eq!(one.trials, 1_712_304, "C(48, 5) boards");
        close(both.share(0), 0.8264, 0.0005, "AA vs KK, both suits shared");
        close(one.share(0), 0.8195, 0.0005, "AA vs KK, one suit shared");
        close(none.share(0), 0.8126, 0.0005, "AA vs KK, no suit shared");
        close(one.share(0) + one.share(1), 1.0, 1e-9, "heads-up shares");

        assert!(
            both.share(0) > one.share(0) && one.share(0) > none.share(0),
            "every shared suit should be worth equity to the aces"
        );
    }

    #[test]
    fn a_range_averages_over_its_suit_configurations() {
        // Six aces combos against six kings combos: one pairing shares both
        // suits, four share one, and one shares none. Averaging the three
        // enumerated figures above must reproduce what sampling the ranges
        // gives, which ties `range_vs_range` back to `exact`.
        let expected = (0.8264 + 4.0 * 0.8195 + 0.8126) / 6.0;
        let ranges = [Range::parse("AA").unwrap(), Range::parse("KK").unwrap()];
        let sampled = range_vs_range(&ranges, &[], &[], 400_000, 17);
        close(sampled.share(0), expected, 0.003, "AA range vs KK range");
    }

    #[test]
    fn monte_carlo_converges_on_the_exact_answer() {
        // Flop, so the exact side is only C(45, 2) = 990 boards.
        let hands = [hand("AsAh"), hand("KsKd")];
        let b = board("2s7h9c");
        let truth = exact(&hands, &b, &[]);
        assert_eq!(truth.trials, 990);
        let sampled = monte_carlo(&hands, &b, &[], 200_000, 99);
        close(sampled.share(0), truth.share(0), 0.005, "sampled vs exact");
    }

    #[test]
    fn the_seed_is_the_only_source_of_randomness() {
        let hands = [hand("AsAh"), hand("KsKd")];
        let a = monte_carlo(&hands, &[], &[], 5_000, 7);
        let b = monte_carlo(&hands, &[], &[], 5_000, 7);
        let c = monte_carlo(&hands, &[], &[], 5_000, 8);
        assert_eq!(a, b, "same seed must replay exactly");
        assert_ne!(a, c, "different seeds must not");
    }

    #[test]
    fn dead_cards_change_the_answer() {
        // Seat 1 holds one card to a flush draw on a two-spade flop. Burning
        // the two remaining spades as dead cards kills the draw outright, so
        // its equity has to fall.
        let hands = [hand("AcAd"), hand("Ks9s")];
        let b = board("2s7s9c");
        let live = exact(&hands, &b, &[]);
        let choked = exact(&hands, &b, &parse_many("4s6s").unwrap());
        assert!(
            choked.share(1) < live.share(1) - 0.02,
            "removing the case spades should cost seat 1 real equity: {:.4} vs {:.4}",
            choked.share(1),
            live.share(1)
        );
    }

    #[test]
    fn range_notation_expands_to_the_right_combo_counts() {
        let n = |s: &str| Range::parse(s).unwrap().len();
        assert_eq!(n("AA"), 6);
        assert_eq!(n("AKs"), 4);
        assert_eq!(n("AKo"), 12);
        assert_eq!(n("AK"), 16);
        assert_eq!(n("AsKh"), 1);
        assert_eq!(n("TT+"), 30, "TT JJ QQ KK AA");
        assert_eq!(n("AJs+"), 12, "AJs AQs AKs");
        assert_eq!(n("77-99"), 18, "77 88 99");
        assert_eq!(n("AJs-A9s"), 12, "A9s ATs AJs");
        assert_eq!(n("AA, KK"), 12);
        assert_eq!(Range::all().len(), 1326);
    }

    #[test]
    fn range_notation_is_order_and_duplicate_insensitive() {
        assert_eq!(Range::parse("AA, AA").unwrap().len(), 6, "duplicates collapse");
        assert_eq!(Range::parse("AKs").unwrap(), Range::parse("KAs").unwrap());
        assert_eq!(Range::parse("AA,KK").unwrap(), Range::parse("KK AA").unwrap());
        assert_eq!(Range::parse("99-77").unwrap(), Range::parse("77-99").unwrap());
        // An explicit combo is just a one-combo range and merges with a class.
        assert_eq!(Range::parse("AA, AsAh").unwrap().len(), 6);
    }

    #[test]
    fn range_notation_rejects_nonsense() {
        for bad in ["AAs", "AAo", "XX", "A", "AKx", "T9s-76s", "AA-AKs", "AsAs"] {
            assert!(Range::parse(bad).is_err(), "`{bad}` should not parse");
        }
    }

    #[test]
    fn card_removal_prunes_a_range() {
        let aces = Range::parse("AA").unwrap();
        assert_eq!(aces.without(&[parse("As").unwrap()]).len(), 3);
        assert_eq!(aces.without(&parse_many("AsAh").unwrap()).len(), 1);
        assert_eq!(aces.without(&parse_many("AsAhAd").unwrap()).len(), 0);
        // A card that is not in the range changes nothing.
        assert_eq!(aces.without(&[parse("2c").unwrap()]).len(), 6);
    }

    #[test]
    fn single_combo_ranges_reduce_to_specific_hands() {
        let b = board("2s7h9c");
        let direct = monte_carlo(&[hand("AsAh"), hand("KsKd")], &b, &[], 100_000, 5);
        let viaranges = range_vs_range(
            &[Range::parse("AsAh").unwrap(), Range::parse("KsKd").unwrap()],
            &b,
            &[],
            100_000,
            5,
        );
        close(viaranges.share(0), direct.share(0), 0.006, "range path vs hand path");
    }

    #[test]
    fn rejection_sampling_matches_exact_enumeration() {
        // The real test of the restart-on-conflict rule. Sampling only the
        // clashing player instead would skew these ranges against each other,
        // because every ace in one range blocks combos in the other.
        let ranges = [Range::parse("AA").unwrap(), Range::parse("AKs, KK").unwrap()];
        let b = board("2s7h9cTd");
        let truth = range_vs_range_exact(&ranges, &b, &[]);
        let sampled = range_vs_range(&ranges, &b, &[], 300_000, 11);
        close(sampled.share(0), truth.share(0), 0.005, "sampled vs enumerated");
    }

    #[test]
    fn identical_ranges_split_evenly() {
        let ranges = [Range::parse("AA").unwrap(), Range::parse("AA").unwrap()];
        let eq = range_vs_range(&ranges, &[], &[], 20_000, 3);
        close(eq.share(0), 0.5, 0.02, "symmetric ranges");
    }

    #[test]
    fn ranges_that_cannot_be_dealt_together_are_an_error() {
        let ranges = [Range::parse("AsAh").unwrap(), Range::parse("AsAh").unwrap()];
        let boom = std::panic::catch_unwind(|| range_vs_range(&ranges, &[], &[], 1, 1));
        assert!(boom.is_err(), "two players cannot both hold AsAh");
    }
}
