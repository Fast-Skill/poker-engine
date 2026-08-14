//! Hand evaluator for 5-, 6-, and 7-card hands.
//!
//! `eval` returns a `u32` where larger is stronger, so comparison is a single
//! integer compare. Layout, high bits first:
//!
//! ```text
//!   bits 20..24   category (0 = high card .. 8 = straight flush)
//!   bits 16..20   primary rank
//!   bits 12..16   secondary rank
//!   bits  8..12   tertiary
//!   bits  4.. 8   quaternary
//!   bits  0.. 4   quinary
//! ```
//!
//! Ranks are 0 (deuce) .. 12 (ace). Unused kicker slots stay zero, which is
//! safe because two hands are only ever compared meaningfully within the same
//! category, and every hand in a given category fills the same slots.
//!
//! This is a branch-and-count evaluator, not a table lookup. It needs no
//! precomputed data and no startup cost, which keeps it easy to verify. When
//! the solver's inner loop makes evaluation the bottleneck, the drop-in
//! upgrade is a perfect-hash or two-plus-two style table behind this exact
//! signature — the rest of the codebase does not need to change.

pub const HIGH_CARD: u32 = 0;
pub const PAIR: u32 = 1;
pub const TWO_PAIR: u32 = 2;
pub const TRIPS: u32 = 3;
pub const STRAIGHT: u32 = 4;
pub const FLUSH: u32 = 5;
pub const FULL_HOUSE: u32 = 6;
pub const QUADS: u32 = 7;
pub const STRAIGHT_FLUSH: u32 = 8;

pub const CATEGORY_NAMES: [&str; 9] = [
    "high card",
    "pair",
    "two pair",
    "three of a kind",
    "straight",
    "flush",
    "full house",
    "four of a kind",
    "straight flush",
];

#[inline]
pub fn category(score: u32) -> u32 {
    score >> 20
}

pub fn describe(score: u32) -> String {
    CATEGORY_NAMES[category(score) as usize].to_string()
}

#[inline]
fn pack(cat: u32, r: [u32; 5]) -> u32 {
    (cat << 20) | (r[0] << 16) | (r[1] << 12) | (r[2] << 8) | (r[3] << 4) | r[4]
}

/// Bits for A-5-4-3-2: the ace plus the four lowest ranks.
const WHEEL: u16 = (1 << 12) | 0b1111;

/// Highest card of the best straight in `mask`, or `None`.
/// The wheel reports rank 3 (a five), which is correct: it is the weakest
/// straight and must lose to a six-high.
#[inline]
fn straight_high(mask: u16) -> Option<u32> {
    let mut hi: i32 = 12;
    while hi >= 4 {
        let window = 0b11111u16 << (hi - 4);
        if mask & window == window {
            return Some(hi as u32);
        }
        hi -= 1;
    }
    if mask & WHEEL == WHEEL {
        return Some(3);
    }
    None
}

/// Write the `n` highest ranks set in `mask` into `out[at..at+n]`.
#[inline]
fn take_top(mask: u16, n: usize, out: &mut [u32; 5], at: usize) {
    let mut slot = at;
    let mut r: i32 = 12;
    while r >= 0 && slot < at + n {
        if mask & (1 << r) != 0 {
            out[slot] = r as u32;
            slot += 1;
        }
        r -= 1;
    }
}

/// Evaluate 5, 6, or 7 cards. Larger is stronger.
pub fn eval(cards: &[u8]) -> u32 {
    debug_assert!(
        (5..=7).contains(&cards.len()),
        "eval takes 5 to 7 cards, got {}",
        cards.len()
    );

    let mut counts = [0u8; 13];
    let mut suits = [0u16; 4];
    let mut ranks = 0u16;

    for &c in cards {
        let r = (c >> 2) as usize;
        let s = (c & 3) as usize;
        counts[r] += 1;
        suits[s] |= 1 << r;
        ranks |= 1 << r;
    }

    // A flush rules out quads and a full house at these hand sizes. The flush
    // consumes five distinct ranks, leaving at most two other cards — not
    // enough for a second rank to appear three or four times. So returning
    // early here is correct even though quads and full houses outrank a flush.
    // Do not "fix" this into a fallthrough without redoing that argument.
    for &fm in &suits {
        if fm.count_ones() >= 5 {
            if let Some(hi) = straight_high(fm) {
                return pack(STRAIGHT_FLUSH, [hi, 0, 0, 0, 0]);
            }
            let mut r = [0u32; 5];
            take_top(fm, 5, &mut r, 0);
            return pack(FLUSH, r);
        }
    }

    // Collect ranks by multiplicity, highest rank first.
    let mut quad: i32 = -1;
    let mut trips = [-1i32; 2];
    let mut n_trips = 0usize;
    let mut pairs = [-1i32; 3];
    let mut n_pairs = 0usize;
    for r in (0..13usize).rev() {
        match counts[r] {
            4 => quad = r as i32,
            3 => {
                trips[n_trips] = r as i32;
                n_trips += 1;
            }
            2 => {
                pairs[n_pairs] = r as i32;
                n_pairs += 1;
            }
            _ => {}
        }
    }

    if quad >= 0 {
        let mut r = [0u32; 5];
        r[0] = quad as u32;
        take_top(ranks & !(1 << quad), 1, &mut r, 1);
        return pack(QUADS, r);
    }

    if n_trips > 0 && (n_trips > 1 || n_pairs > 0) {
        // With two sets of trips the lower set plays as the pair.
        let pair = if n_trips > 1 && (n_pairs == 0 || trips[1] > pairs[0]) {
            trips[1] as u32
        } else {
            pairs[0] as u32
        };
        return pack(FULL_HOUSE, [trips[0] as u32, pair, 0, 0, 0]);
    }

    // Straights are checked after quads and full houses only for ordering
    // clarity — the same counting argument shows they cannot co-occur.
    // Checking before trips and two pair is load-bearing: those *can*
    // co-occur with a straight, and the straight outranks them.
    if let Some(hi) = straight_high(ranks) {
        return pack(STRAIGHT, [hi, 0, 0, 0, 0]);
    }

    if n_trips > 0 {
        let mut r = [0u32; 5];
        r[0] = trips[0] as u32;
        take_top(ranks & !(1 << trips[0]), 2, &mut r, 1);
        return pack(TRIPS, r);
    }

    if n_pairs >= 2 {
        let mut r = [0u32; 5];
        r[0] = pairs[0] as u32;
        r[1] = pairs[1] as u32;
        // With three pairs the third pair's rank is still a live kicker.
        let used = (1u16 << pairs[0]) | (1u16 << pairs[1]);
        take_top(ranks & !used, 1, &mut r, 2);
        return pack(TWO_PAIR, r);
    }

    if n_pairs == 1 {
        let mut r = [0u32; 5];
        r[0] = pairs[0] as u32;
        take_top(ranks & !(1 << pairs[0]), 3, &mut r, 1);
        return pack(PAIR, r);
    }

    let mut r = [0u32; 5];
    take_top(ranks, 5, &mut r, 0);
    pack(HIGH_CARD, r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_many;

    fn e(s: &str) -> u32 {
        eval(&parse_many(s).unwrap())
    }

    #[test]
    fn categories_are_detected() {
        assert_eq!(category(e("AsKsQsJsTs")), STRAIGHT_FLUSH);
        assert_eq!(category(e("5s4s3s2sAs")), STRAIGHT_FLUSH);
        assert_eq!(category(e("7c7d7h7s2c")), QUADS);
        assert_eq!(category(e("7c7d7hKsKc")), FULL_HOUSE);
        assert_eq!(category(e("As9s7s4s2s")), FLUSH);
        assert_eq!(category(e("9c8d7h6s5c")), STRAIGHT);
        assert_eq!(category(e("5c4d3h2sAc")), STRAIGHT);
        assert_eq!(category(e("7c7d7hKs2c")), TRIPS);
        assert_eq!(category(e("7c7dKhKs2c")), TWO_PAIR);
        assert_eq!(category(e("7c7dKhQs2c")), PAIR);
        assert_eq!(category(e("Ac9d7h4s2c")), HIGH_CARD);
    }

    #[test]
    fn category_ordering_holds() {
        let ladder = [
            e("Ac9d7h4s2c"), // high card
            e("7c7dKhQs2c"), // pair
            e("7c7dKhKs2c"), // two pair
            e("7c7d7hKs2c"), // trips
            e("9c8d7h6s5c"), // straight
            e("As9s7s4s2s"), // flush
            e("7c7d7hKsKc"), // full house
            e("7c7d7h7s2c"), // quads
            e("AsKsQsJsTs"), // straight flush
        ];
        for w in ladder.windows(2) {
            assert!(w[0] < w[1], "{:#x} should rank below {:#x}", w[0], w[1]);
        }
    }

    #[test]
    fn wheel_is_the_weakest_straight() {
        assert!(e("5c4d3h2sAc") < e("6c5d4h3s2c"));
        assert!(e("5s4s3s2sAs") < e("6s5s4s3s2s"));
    }

    #[test]
    fn ace_high_straight_beats_king_high() {
        assert!(e("AcKdQhJsTc") > e("KcQdJhTs9c"));
    }

    #[test]
    fn seven_cards_pick_the_best_five() {
        // Board pairs the board; the flush is still the hand.
        assert_eq!(category(e("As9s7s4s2s 8h8d")), FLUSH);
        // Straight present alongside a pair — the straight plays.
        assert_eq!(category(e("9c8d7h6s5c 5d2h")), STRAIGHT);
        // Two sets of trips make a full house, higher set on top.
        let h = e("7c7d7h 5s5d5c 2h");
        assert_eq!(category(h), FULL_HOUSE);
        assert_eq!(h, e("7c7d7h5s5d2h3c"));
    }

    #[test]
    fn kickers_matter() {
        assert!(e("7c7dAh4s2c") > e("7c7dKh4s2c"));
        assert!(e("7c7d7hAs2c") > e("7c7d7hKs2c"));
        assert!(e("Ac7d7hKsKc") > e("2c7d7hKsKc"));
        // Third pair supplies the kicker in a three-pair seven-card hand.
        assert!(e("AcAdKcKd9c9d2h") > e("AcAdKcKd8c8d2h"));
    }

    #[test]
    fn suits_never_break_ties() {
        assert_eq!(e("AcKd9h7s5c"), e("AsKh9d7c5s"));
    }

    #[test]
    fn quads_beat_a_lower_full_house_and_flushes() {
        assert!(e("2c2d2h2sAc") > e("AcAdAhKsKc"));
        assert!(e("2c2d2h2sAc") > e("AsQsTs8s6s"));
    }

    #[test]
    fn flush_uses_its_five_best() {
        // Six spades: the deuce must be dropped.
        assert_eq!(e("AsKs9s7s5s2s 3h"), e("AsKs9s7s5s 3h4d"));
    }

    #[test]
    fn every_seven_card_hand_is_evaluable() {
        // Exhaustive-ish smoke test: walk a lot of distinct 7-card sets and
        // confirm nothing panics and every score decodes to a real category.
        let mut n = 0u64;
        for a in 0..52u8 {
            for b in (a + 1)..52u8 {
                let cards = [a, b, (a + 7) % 52, (b + 13) % 52, (a + 23) % 52, (b + 31) % 52, (a + 41) % 52];
                let mut seen = [false; 52];
                if cards.iter().any(|&c| std::mem::replace(&mut seen[c as usize], true)) {
                    continue; // skip generated duplicates
                }
                let s = eval(&cards);
                assert!(category(s) <= STRAIGHT_FLUSH);
                n += 1;
            }
        }
        assert!(n > 500, "expected a meaningful number of samples, got {n}");
    }
}
