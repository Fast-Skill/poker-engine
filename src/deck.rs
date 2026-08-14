//! A seeded 52-card deck.
//!
//! The RNG is xoshiro256++ inlined rather than pulled from a crate, so the
//! shuffle is reproducible across machines and toolchain versions. Solver runs
//! and variance-reduction work both depend on being able to replay an exact
//! sequence of hands, so the seed is always explicit — there is no
//! `from_entropy` convenience here on purpose.

pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        // SplitMix64 to spread a single seed word across the full state.
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng { s: [next(), next(), next(), next()] }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0].wrapping_add(self.s[3]).rotate_left(23).wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `0..n`, rejection-sampled so there is no modulo bias.
    /// Bias matters here: a skewed shuffle quietly corrupts every equity
    /// number downstream, and it is invisible without looking for it.
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        let zone = u64::MAX - (u64::MAX % n) - 1;
        loop {
            let x = self.next_u64();
            if x <= zone {
                return x % n;
            }
        }
    }
}

pub struct Deck {
    cards: [u8; 52],
    dealt: usize,
    rng: Rng,
}

impl Deck {
    pub fn new(seed: u64) -> Deck {
        let mut cards = [0u8; 52];
        for (i, c) in cards.iter_mut().enumerate() {
            *c = i as u8;
        }
        Deck { cards, dealt: 0, rng: Rng::new(seed) }
    }

    /// Fisher-Yates, and reset the deal pointer.
    pub fn shuffle(&mut self) {
        for i in (1..52).rev() {
            let j = self.rng.below(i as u64 + 1) as usize;
            self.cards.swap(i, j);
        }
        self.dealt = 0;
    }

    #[inline]
    pub fn deal(&mut self, n: usize) -> &[u8] {
        assert!(self.dealt + n <= 52, "deck exhausted");
        let start = self.dealt;
        self.dealt += n;
        &self.cards[start..self.dealt]
    }

    #[inline]
    pub fn deal_one(&mut self) -> u8 {
        self.deal(1)[0]
    }

    pub fn remaining(&self) -> usize {
        52 - self.dealt
    }

    /// Remove specific cards from the undealt portion — used when a hand is
    /// partially specified (known hole cards, a fixed board) and the rest
    /// still needs to run out randomly.
    pub fn remove(&mut self, cards: &[u8]) {
        for &c in cards {
            if let Some(pos) = self.cards[self.dealt..].iter().position(|&x| x == c) {
                let abs = self.dealt + pos;
                self.cards.swap(self.dealt, abs);
                self.dealt += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_is_a_permutation() {
        let mut d = Deck::new(12345);
        d.shuffle();
        let mut seen = [false; 52];
        for &c in d.deal(52) {
            assert!(c < 52);
            assert!(!seen[c as usize], "card {c} dealt twice");
            seen[c as usize] = true;
        }
    }

    #[test]
    fn same_seed_same_shuffle() {
        let mut a = Deck::new(99);
        let mut b = Deck::new(99);
        a.shuffle();
        b.shuffle();
        assert_eq!(a.deal(52), b.deal(52));
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = Deck::new(1);
        let mut b = Deck::new(2);
        a.shuffle();
        b.shuffle();
        assert_ne!(a.deal(52), b.deal(52));
    }

    #[test]
    fn remove_takes_cards_out_of_play() {
        let mut d = Deck::new(7);
        d.shuffle();
        let blocked = [0u8, 51, 25];
        d.remove(&blocked);
        assert_eq!(d.remaining(), 49);
        for &c in d.deal(49) {
            assert!(!blocked.contains(&c), "removed card {c} was still dealt");
        }
    }

    #[test]
    fn below_stays_in_range() {
        let mut r = Rng::new(4242);
        for n in 1..=52u64 {
            for _ in 0..200 {
                assert!(r.below(n) < n);
            }
        }
    }
}
