//! No-limit hold'em betting engine.
//!
//! The engine owns pot arithmetic, legal-action generation, the min-raise and
//! reopening rules, side pots, and showdown payouts. It deliberately does not
//! deal — the caller supplies board cards. Keeping dealing outside means a
//! hand can be replayed exactly, which is what tests and any
//! variance-reduction work need.
//!
//! Chips are integers in the smallest unit you care about. Do not use floats
//! for money.

use crate::card::NO_CARD;
use crate::eval::eval;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Active,
    Folded,
    AllIn,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Street {
    Preflop,
    Flop,
    Turn,
    River,
    Showdown,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    Fold,
    Check,
    Call,
    /// Raise the street commitment **to** this total, not **by** it.
    /// "Raise to" is the only unambiguous encoding once short all-ins and
    /// blinds are in play, so the whole engine speaks it.
    RaiseTo(u32),
}

/// What the player to act may legally do.
#[derive(Clone, Copy, Debug)]
pub struct Legal {
    pub fold: bool,
    pub check: bool,
    /// Chips required to call, already capped at the player's stack.
    pub call: Option<u32>,
    /// `(minimum raise-to, maximum raise-to)`, inclusive. When a short stack
    /// cannot make a full raise the two are equal — the all-in.
    pub raise: Option<(u32, u32)>,
}

#[derive(Clone, Debug)]
pub struct Player {
    pub stack: u32,
    pub hole: [u8; 2],
    /// Total put in across the whole hand — the basis for side pots.
    pub committed: u32,
    /// Put in on the current street — the basis for what is owed.
    pub street_committed: u32,
    pub status: Status,
    /// Has acted since the last full raise. Cleared for everyone when the
    /// action reopens.
    pub acted: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pot {
    pub amount: u32,
    pub eligible: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Game {
    pub players: Vec<Player>,
    pub board: Vec<u8>,
    pub street: Street,
    pub button: usize,
    pub to_act: usize,
    /// Highest street commitment anyone has made this street.
    pub bet: u32,
    /// Size of the last full raise increment; the min-raise is `bet + this`.
    pub last_raise: u32,
    pub bb: u32,
}

impl Game {
    /// Start a hand and post the blinds.
    ///
    /// Heads-up, the button posts the small blind and acts first preflop but
    /// last on every later street. That inversion is a real rule, not an
    /// off-by-one.
    pub fn new(stacks: &[u32], button: usize, sb: u32, bb: u32) -> Game {
        let n = stacks.len();
        assert!(n >= 2, "need at least two players");
        assert!(button < n, "button seat out of range");
        assert!(sb <= bb && bb > 0, "blinds must be positive with sb <= bb");

        let players = stacks
            .iter()
            .map(|&s| Player {
                stack: s,
                hole: [NO_CARD; 2],
                committed: 0,
                street_committed: 0,
                status: Status::Active,
                acted: false,
            })
            .collect();

        let (sb_pos, bb_pos) = if n == 2 {
            (button, (button + 1) % n)
        } else {
            ((button + 1) % n, (button + 2) % n)
        };

        let mut g = Game {
            players,
            board: Vec::with_capacity(5),
            street: Street::Preflop,
            button,
            to_act: 0,
            bet: bb,
            last_raise: bb,
            bb,
        };
        g.post(sb_pos, sb);
        g.post(bb_pos, bb);
        // The blinds are forced, not voluntary: the big blind still has the
        // option to raise, so nobody is marked as having acted.
        g.to_act = g.next_active_from((bb_pos + 1) % n);
        g
    }

    pub fn set_hole(&mut self, seat: usize, cards: [u8; 2]) {
        self.players[seat].hole = cards;
    }

    fn post(&mut self, seat: usize, amount: u32) {
        self.pay(seat, amount);
    }

    fn pay(&mut self, seat: usize, amount: u32) {
        let p = &mut self.players[seat];
        let paid = amount.min(p.stack);
        p.stack -= paid;
        p.committed += paid;
        p.street_committed += paid;
        if p.stack == 0 {
            p.status = Status::AllIn;
        }
    }

    fn next_active_from(&self, start: usize) -> usize {
        let n = self.players.len();
        for k in 0..n {
            let i = (start + k) % n;
            if self.players[i].status == Status::Active {
                return i;
            }
        }
        start
    }

    pub fn pot(&self) -> u32 {
        self.players.iter().map(|p| p.committed).sum()
    }

    pub fn live_seats(&self) -> Vec<usize> {
        (0..self.players.len())
            .filter(|&i| self.players[i].status != Status::Folded)
            .collect()
    }

    /// Non-folded seats, counted without building the `Vec` that `live_seats`
    /// returns. `betting_done` runs on every `apply`, so the allocation showed
    /// up directly in the engine benchmark.
    fn live_count(&self) -> usize {
        self.players.iter().filter(|p| p.status != Status::Folded).count()
    }

    /// True when a player other than the one to act still has chips behind.
    ///
    /// A raise needs someone able to call it. With every opponent already
    /// all-in the extra chips can only return to the raiser as an uncalled bet
    /// at showdown, so the action is not legal at a table and it hangs a dead
    /// subtree off the solver's game tree.
    fn others_can_act(&self) -> bool {
        self.players
            .iter()
            .enumerate()
            .any(|(i, q)| i != self.to_act && q.status == Status::Active)
    }

    pub fn legal(&self) -> Legal {
        let p = &self.players[self.to_act];
        let owed = self.bet.saturating_sub(p.street_committed);
        let max_to = p.street_committed + p.stack;

        // Three conditions gate a raise. You need chips to get past the current
        // bet; you need the right to raise at all, since a player who has
        // already acted only regains it when someone makes a full raise, which
        // is what clears `acted` — facing a short all-in you may call or fold,
        // never re-raise; and somebody else must still have chips to call with.
        // `others_can_act` is last so it is only evaluated when the cheap tests
        // have already passed.
        let raise = if max_to > self.bet && !p.acted && self.others_can_act() {
            let min_to = (self.bet + self.last_raise).min(max_to);
            Some((min_to, max_to))
        } else {
            None
        };

        Legal {
            // Folding when you could check for free is legal at a table but
            // never correct, and keeping it out halves nothing while removing
            // a whole class of dominated branches from the tree.
            fold: owed > 0,
            check: owed == 0,
            call: if owed > 0 { Some(owed.min(p.stack)) } else { None },
            raise,
        }
    }

    pub fn apply(&mut self, action: Action) {
        assert!(!self.betting_done(), "betting round is already complete");
        let seat = self.to_act;
        let legal = self.legal();

        match action {
            Action::Fold => {
                assert!(legal.fold, "cannot fold with nothing owed");
                self.players[seat].status = Status::Folded;
            }
            Action::Check => {
                assert!(legal.check, "cannot check facing a bet");
            }
            Action::Call => {
                let amount = legal.call.expect("nothing to call");
                self.pay(seat, amount);
            }
            Action::RaiseTo(to) => {
                let (min_to, max_to) = legal.raise.expect("cannot raise");
                assert!(
                    to >= min_to && to <= max_to,
                    "raise to {to} outside legal band {min_to}..={max_to}"
                );
                let add = to - self.players[seat].street_committed;
                self.pay(seat, add);

                let increment = to - self.bet;
                // Only a full-size raise reopens the action. An all-in for
                // less than a full raise increases the bet but does *not* give
                // players who already acted the right to raise again. Getting
                // this wrong silently inflates the game tree and produces
                // strategies that are not legal at a real table.
                if increment >= self.last_raise {
                    self.last_raise = increment;
                    for (j, q) in self.players.iter_mut().enumerate() {
                        if j != seat && q.status == Status::Active {
                            q.acted = false;
                        }
                    }
                }
                self.bet = to;
            }
        }

        self.players[seat].acted = true;
        if !self.betting_done() {
            self.to_act = self.next_active_from((seat + 1) % self.players.len());
        }
    }

    pub fn betting_done(&self) -> bool {
        if self.live_count() <= 1 {
            return true;
        }

        // One pass over the seats, no allocation.
        let mut n_active = 0usize;
        let mut lone = 0usize;
        let mut all_matched = true;
        for (i, p) in self.players.iter().enumerate() {
            if p.status != Status::Active {
                continue;
            }
            n_active += 1;
            lone = i;
            if !p.acted || p.street_committed != self.bet {
                all_matched = false;
            }
        }

        match n_active {
            // Everyone is all-in; nothing left to decide.
            0 => true,
            // One player still has chips and nobody else can act, so the street
            // ends the moment they owe nothing. This deliberately does *not*
            // require `acted`: every new street clears that flag, and demanding
            // it here would hand the last live player a betting round on each
            // remaining street with no opponent able to call.
            1 => self.players[lone].street_committed >= self.bet,
            _ => all_matched,
        }
    }

    /// True when the hand is over: everyone folded but one, or the river
    /// betting is complete.
    pub fn hand_done(&self) -> bool {
        self.live_count() <= 1
            || (self.street == Street::River && self.betting_done())
            || self.street == Street::Showdown
    }

    /// Advance to the next street with the supplied board cards (3, then 1,
    /// then 1).
    pub fn next_street(&mut self, cards: &[u8]) {
        assert!(self.betting_done(), "betting round is not complete");
        assert!(!self.hand_done(), "hand is already over");

        let expect = match self.street {
            Street::Preflop => 3,
            Street::Flop | Street::Turn => 1,
            Street::River | Street::Showdown => panic!("no street after the river"),
        };
        assert_eq!(cards.len(), expect, "wrong number of board cards");

        self.board.extend_from_slice(cards);
        self.street = match self.street {
            Street::Preflop => Street::Flop,
            Street::Flop => Street::Turn,
            Street::Turn => Street::River,
            _ => unreachable!(),
        };

        self.bet = 0;
        self.last_raise = self.bb;
        for p in self.players.iter_mut() {
            p.street_committed = 0;
            p.acted = false;
        }
        // Postflop the first live seat left of the button acts first, which is
        // why heads-up the button now acts last.
        self.to_act = self.next_active_from((self.button + 1) % self.players.len());
    }

    /// Split the pot into a main pot and any side pots, lowest first.
    ///
    /// Chips from folded players are dead money and belong to the lowest pots
    /// they reached, so they are counted here even though those seats are not
    /// eligible to win.
    pub fn side_pots(&self) -> Vec<Pot> {
        let mut levels: Vec<u32> = self
            .players
            .iter()
            .filter(|p| p.status != Status::Folded && p.committed > 0)
            .map(|p| p.committed)
            .collect();
        levels.sort_unstable();
        levels.dedup();

        let mut pots = Vec::new();
        let mut prev = 0u32;
        for lvl in levels {
            let amount: u32 = self
                .players
                .iter()
                .map(|p| p.committed.min(lvl).saturating_sub(prev))
                .sum();
            let eligible: Vec<usize> = (0..self.players.len())
                .filter(|&i| {
                    self.players[i].status != Status::Folded && self.players[i].committed >= lvl
                })
                .collect();
            if amount > 0 {
                pots.push(Pot { amount, eligible });
            }
            prev = lvl;
        }
        pots
    }

    /// Chips returned to each seat. Indexed by seat; sums to the pot.
    pub fn payouts(&self) -> Vec<u32> {
        let n = self.players.len();
        let mut out = vec![0u32; n];
        let live = self.live_seats();

        if live.len() == 1 {
            out[live[0]] = self.pot();
            return out;
        }

        assert_eq!(self.board.len(), 5, "showdown needs a full board");
        let mut strength = vec![0u32; n];
        for &i in &live {
            let h = self.players[i].hole;
            assert!(h[0] != NO_CARD && h[1] != NO_CARD, "seat {i} has no hole cards");
            let seven = [
                self.board[0], self.board[1], self.board[2], self.board[3], self.board[4], h[0],
                h[1],
            ];
            strength[i] = eval(&seven);
        }

        for pot in self.side_pots() {
            let best = pot.eligible.iter().map(|&i| strength[i]).max().unwrap();
            let mut winners: Vec<usize> =
                pot.eligible.iter().copied().filter(|&i| strength[i] == best).collect();
            // Odd chips go to the first winner left of the button.
            let first = (self.button + 1) % n;
            winners.sort_by_key(|&i| (i + n - first) % n);

            let share = pot.amount / winners.len() as u32;
            let mut odd = pot.amount - share * winners.len() as u32;
            for &w in &winners {
                out[w] += share;
                if odd > 0 {
                    out[w] += 1;
                    odd -= 1;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_many;

    fn holdem(stacks: &[u32]) -> Game {
        Game::new(stacks, 0, 1, 2)
    }

    #[test]
    fn blinds_are_posted_and_action_starts_utg() {
        let g = Game::new(&[100, 100, 100], 0, 1, 2);
        assert_eq!(g.players[1].committed, 1); // small blind
        assert_eq!(g.players[2].committed, 2); // big blind
        assert_eq!(g.to_act, 0); // button is UTG three-handed
        assert_eq!(g.bet, 2);
        assert_eq!(g.pot(), 3);
    }

    #[test]
    fn heads_up_button_posts_small_blind_and_acts_first() {
        let g = holdem(&[100, 100]);
        assert_eq!(g.players[0].committed, 1);
        assert_eq!(g.players[1].committed, 2);
        assert_eq!(g.to_act, 0);
    }

    #[test]
    fn big_blind_keeps_the_option() {
        let mut g = holdem(&[100, 100]);
        g.apply(Action::Call); // button limps
        assert!(!g.betting_done(), "big blind still has the option to raise");
        assert_eq!(g.to_act, 1);
        assert!(g.legal().check);
        assert!(g.legal().raise.is_some());
        g.apply(Action::Check);
        assert!(g.betting_done());
    }

    #[test]
    fn min_raise_tracks_the_last_increment() {
        let mut g = holdem(&[200, 200]);
        assert_eq!(g.legal().raise, Some((4, 200))); // bet 2 + last raise 2
        g.apply(Action::RaiseTo(6));
        assert_eq!(g.legal().raise, Some((10, 200))); // bet 6 + increment 4
        g.apply(Action::RaiseTo(20));
        assert_eq!(g.legal().raise, Some((34, 200))); // bet 20 + increment 14
    }

    #[test]
    fn short_all_in_does_not_reopen_the_action() {
        // Button raises to 20, the short big blind jams for 25. That is a
        // 5-chip increment against a last raise of 18 — not a full raise, so
        // the original raiser may call or fold but must not raise again.
        let mut g = Game::new(&[200, 200, 25], 0, 1, 2);
        g.apply(Action::RaiseTo(20)); // seat 0
        g.apply(Action::Fold); // seat 1
        assert_eq!(g.to_act, 2);
        assert_eq!(g.legal().raise, Some((25, 25)), "short stack can only jam");
        g.apply(Action::RaiseTo(25)); // seat 2 jams short
        assert_eq!(g.players[2].status, Status::AllIn);
        assert_eq!(g.to_act, 0);
        let legal = g.legal();
        assert_eq!(legal.call, Some(5));
        assert!(legal.fold);
        assert!(legal.raise.is_none(), "short all-in must not reopen betting");
        // And the hand still resolves cleanly from there.
        g.apply(Action::Call);
        assert!(g.betting_done());
    }

    #[test]
    fn short_all_in_does_not_reopen_even_with_live_opponents() {
        // Four-handed, so the opener still faces players with chips after the
        // jam. That isolates the reopening rule from the all-opponents-all-in
        // guard below — with only two seats live, either one alone would block
        // the raise and a regression in the other would go unnoticed.
        let mut g = Game::new(&[200, 200, 200, 25], 3, 1, 2);
        assert_eq!(g.to_act, 2);
        g.apply(Action::RaiseTo(20)); // seat 2 opens
        g.apply(Action::RaiseTo(25)); // seat 3 jams, a 5-chip increment
        assert_eq!(g.players[3].status, Status::AllIn);
        g.apply(Action::Call); // seat 0
        g.apply(Action::Call); // seat 1
        assert_eq!(g.to_act, 2, "the opener still owes the difference");

        let legal = g.legal();
        assert_eq!(legal.call, Some(5));
        assert!(
            legal.raise.is_none(),
            "a short all-in must not reopen betting for a player who already acted"
        );
        // Both callers still have chips, so what blocked the raise above is the
        // reopening rule and not the all-in guard.
        assert_eq!(g.players[0].status, Status::Active);
        assert_eq!(g.players[1].status, Status::Active);
    }

    #[test]
    fn cannot_raise_when_every_opponent_is_all_in() {
        // Seat 1 jams for less than seat 0 has behind. Seat 0 has every other
        // right to raise here — the full jam reopened the action — but there is
        // nobody left to call, so raising would only put in chips that come
        // straight back as an uncalled bet.
        let mut g = Game::new(&[200, 40], 0, 1, 2);
        g.apply(Action::RaiseTo(10)); // seat 0 opens
        g.apply(Action::RaiseTo(40)); // seat 1 jams, a full re-raise
        assert_eq!(g.players[1].status, Status::AllIn);
        assert_eq!(g.to_act, 0);

        let legal = g.legal();
        assert_eq!(legal.call, Some(30));
        assert!(legal.fold);
        assert!(
            legal.raise.is_none(),
            "no opponent has chips behind, so there is nothing to raise into"
        );
    }

    #[test]
    fn all_opponents_all_in_ends_every_later_street() {
        // Once seat 1 is all-in and covered, no street has any action left in
        // it. Requiring `acted` here used to reopen a dead betting round on
        // each new street, because `next_street` clears that flag for everyone.
        let mut g = Game::new(&[200, 50], 0, 1, 2);
        g.apply(Action::RaiseTo(50));
        g.apply(Action::Call); // seat 1 calls all-in
        assert_eq!(g.players[1].status, Status::AllIn);
        assert!(g.betting_done());

        for street in ["2c7d9h", "3s", "4d"] {
            g.next_street(&parse_many(street).unwrap());
            assert!(
                g.betting_done(),
                "no opponent can call, so {:?} needs no action",
                g.street
            );
        }
        assert!(g.hand_done());
    }

    #[test]
    fn full_reraise_does_reopen_the_action() {
        let mut g = Game::new(&[200, 200, 200], 0, 1, 2);
        g.apply(Action::RaiseTo(6));
        g.apply(Action::Fold);
        g.apply(Action::RaiseTo(18)); // full re-raise
        assert!(g.legal().raise.is_some());
    }

    #[test]
    fn street_advances_and_resets_the_betting() {
        let mut g = holdem(&[100, 100]);
        g.apply(Action::Call);
        g.apply(Action::Check);
        g.next_street(&parse_many("2c7d9h").unwrap());
        assert_eq!(g.street, Street::Flop);
        assert_eq!(g.bet, 0);
        assert_eq!(g.board.len(), 3);
        // Heads-up the button acts last postflop.
        assert_eq!(g.to_act, 1);
        assert!(g.legal().check);
        assert!(!g.legal().fold);
    }

    #[test]
    fn folding_to_one_player_ends_the_hand() {
        let mut g = holdem(&[100, 100]);
        g.apply(Action::Fold);
        assert!(g.hand_done());
        let pay = g.payouts();
        assert_eq!(pay[1], 3);
        assert_eq!(pay[0], 0);
    }

    #[test]
    fn side_pots_split_at_each_all_in_level() {
        let mut g = Game::new(&[50, 100, 100], 0, 1, 2);
        g.apply(Action::RaiseTo(50)); // seat 0 jams for 50
        g.apply(Action::RaiseTo(100)); // seat 1 jams for 100
        g.apply(Action::Call); // seat 2 calls 100
        let pots = g.side_pots();
        assert_eq!(pots.len(), 2);
        assert_eq!(pots[0].amount, 150); // 50 x 3
        assert_eq!(pots[0].eligible, vec![0, 1, 2]);
        assert_eq!(pots[1].amount, 100); // 50 x 2
        assert_eq!(pots[1].eligible, vec![1, 2]);
        assert_eq!(pots.iter().map(|p| p.amount).sum::<u32>(), g.pot());
    }

    #[test]
    fn dead_money_from_folders_lands_in_the_main_pot() {
        let mut g = Game::new(&[100, 100, 100], 0, 1, 2);
        g.apply(Action::RaiseTo(10)); // seat 0
        g.apply(Action::Call); // seat 1 puts in 10
        g.apply(Action::Fold); // seat 2 forfeits its 2
        let pots = g.side_pots();
        assert_eq!(pots.len(), 1);
        assert_eq!(pots[0].amount, 22, "the folded big blind is dead money");
        assert_eq!(pots[0].eligible, vec![0, 1]);
    }

    #[test]
    fn showdown_pays_the_best_hand() {
        let mut g = holdem(&[100, 100]);
        g.set_hole(0, [crate::card::parse("As").unwrap(), crate::card::parse("Ah").unwrap()]);
        g.set_hole(1, [crate::card::parse("Kc").unwrap(), crate::card::parse("Kd").unwrap()]);
        g.apply(Action::Call);
        g.apply(Action::Check);
        g.next_street(&parse_many("2c7d9h").unwrap());
        g.apply(Action::Check);
        g.apply(Action::Check);
        g.next_street(&parse_many("3s").unwrap());
        g.apply(Action::Check);
        g.apply(Action::Check);
        g.next_street(&parse_many("4d").unwrap());
        g.apply(Action::Check);
        g.apply(Action::Check);
        assert!(g.hand_done());
        let pay = g.payouts();
        assert_eq!(pay[0], 4, "aces win the four-chip pot");
        assert_eq!(pay[1], 0);
    }

    #[test]
    fn split_pot_is_even_and_conserves_chips() {
        let mut g = holdem(&[100, 100]);
        g.set_hole(0, [crate::card::parse("As").unwrap(), crate::card::parse("Kh").unwrap()]);
        g.set_hole(1, [crate::card::parse("Ad").unwrap(), crate::card::parse("Kc").unwrap()]);
        g.apply(Action::Call);
        g.apply(Action::Check);
        g.next_street(&parse_many("2c7d9h").unwrap());
        g.apply(Action::Check);
        g.apply(Action::Check);
        g.next_street(&parse_many("3s").unwrap());
        g.apply(Action::Check);
        g.apply(Action::Check);
        g.next_street(&parse_many("4d").unwrap());
        g.apply(Action::Check);
        g.apply(Action::Check);
        let pay = g.payouts();
        assert_eq!(pay, vec![2, 2]);
        assert_eq!(pay.iter().sum::<u32>(), g.pot());
    }

    #[test]
    fn chips_are_conserved_across_a_random_walk() {
        // Play many hands with arbitrary legal actions and assert no chips are
        // created or destroyed. Conservation is the invariant most likely to
        // catch a subtle pot-arithmetic bug.
        let mut rng = crate::deck::Rng::new(2024);
        let mut deck = crate::deck::Deck::new(7);
        for _ in 0..2000 {
            deck.shuffle();
            let stacks = [
                20 + rng.below(180) as u32,
                20 + rng.below(180) as u32,
                20 + rng.below(180) as u32,
            ];
            let total: u32 = stacks.iter().sum();
            let mut g = Game::new(&stacks, rng.below(3) as usize, 1, 2);
            for seat in 0..3 {
                let c = deck.deal(2);
                g.set_hole(seat, [c[0], c[1]]);
            }

            while !g.hand_done() {
                if g.betting_done() {
                    let n = if g.street == Street::Preflop { 3 } else { 1 };
                    let c: Vec<u8> = deck.deal(n).to_vec();
                    g.next_street(&c);
                    continue;
                }
                let legal = g.legal();
                let mut choices: Vec<Action> = Vec::new();
                if legal.check {
                    choices.push(Action::Check);
                }
                if legal.call.is_some() {
                    choices.push(Action::Call);
                }
                if legal.fold {
                    choices.push(Action::Fold);
                }
                if let Some((lo, hi)) = legal.raise {
                    let to = lo + rng.below((hi - lo + 1) as u64) as u32;
                    choices.push(Action::RaiseTo(to));
                }
                assert!(!choices.is_empty(), "player to act had no legal action");
                let pick = choices[rng.below(choices.len() as u64) as usize];
                g.apply(pick);
            }

            // A hand that reaches showdown needs a full board to be paid out.
            while g.live_seats().len() > 1 && g.board.len() < 5 {
                let n = if g.board.is_empty() { 3 } else { 1 };
                let c: Vec<u8> = deck.deal(n).to_vec();
                g.board.extend_from_slice(&c);
            }

            let pay = g.payouts();
            assert_eq!(pay.iter().sum::<u32>(), g.pot(), "pot did not balance");
            let final_total: u32 =
                g.players.iter().map(|p| p.stack).sum::<u32>() + pay.iter().sum::<u32>();
            assert_eq!(final_total, total, "chips were created or destroyed");
        }
    }
}
