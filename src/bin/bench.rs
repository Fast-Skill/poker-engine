//! Throughput check for the evaluator and the betting engine.
//!
//! Run with `cargo run --release --bin bench`. Debug builds are roughly an
//! order of magnitude slower and the number means nothing there.

use std::time::Instant;

use poker_engine::card;
use poker_engine::deck::{Deck, Rng};
use poker_engine::equity;
use poker_engine::eval::eval;
use poker_engine::game::{Action, Game, Street};

fn bench_eval() {
    // Pre-generate hands so the timed loop measures evaluation, not shuffling.
    let mut deck = Deck::new(0xC0FFEE);
    let n = 1_000_000usize;
    let mut hands: Vec<[u8; 7]> = Vec::with_capacity(n);
    for _ in 0..n {
        deck.shuffle();
        let c = deck.deal(7);
        hands.push([c[0], c[1], c[2], c[3], c[4], c[5], c[6]]);
    }

    let start = Instant::now();
    let mut checksum = 0u64;
    for h in &hands {
        checksum = checksum.wrapping_add(eval(h) as u64);
    }
    let dt = start.elapsed();

    let per_sec = n as f64 / dt.as_secs_f64();
    println!(
        "eval:   {:>10} hands in {:>8.3?}  =>  {:>8.2} M/s   (checksum {checksum})",
        n,
        dt,
        per_sec / 1e6
    );
}

fn bench_engine() {
    let n = 200_000usize;
    let mut rng = Rng::new(1);
    let mut deck = Deck::new(2);
    let start = Instant::now();
    let mut chips = 0u64;

    for i in 0..n {
        deck.shuffle();
        let mut g = Game::new(&[200, 200, 200, 200, 200, 200], i % 6, 1, 2);
        for seat in 0..6 {
            let c = deck.deal(2);
            g.set_hole(seat, [c[0], c[1]]);
        }
        while !g.hand_done() {
            if g.betting_done() {
                let k = if g.street == Street::Preflop { 3 } else { 1 };
                let c: Vec<u8> = deck.deal(k).to_vec();
                g.next_street(&c);
                continue;
            }
            let legal = g.legal();
            // A crude mix: mostly check/call, occasional raise, occasional fold.
            let roll = rng.below(100);
            let action = if roll < 8 {
                match legal.raise {
                    Some((lo, hi)) => Action::RaiseTo(lo + rng.below((hi - lo + 1) as u64) as u32),
                    None if legal.check => Action::Check,
                    None => Action::Call,
                }
            } else if roll < 20 && legal.fold {
                Action::Fold
            } else if legal.check {
                Action::Check
            } else {
                Action::Call
            };
            g.apply(action);
        }
        while g.live_seats().len() > 1 && g.board.len() < 5 {
            let k = if g.board.is_empty() { 3 } else { 1 };
            let c: Vec<u8> = deck.deal(k).to_vec();
            g.board.extend_from_slice(&c);
        }
        chips += g.payouts().iter().map(|&x| x as u64).sum::<u64>();
    }

    let dt = start.elapsed();
    let per_sec = n as f64 / dt.as_secs_f64();
    println!(
        "engine: {:>10} hands in {:>8.3?}  =>  {:>8.2} K/s   (chips paid {chips})",
        n,
        dt,
        per_sec / 1e3
    );
}

fn bench_equity() {
    let aa = [card::parse("As").unwrap(), card::parse("Ah").unwrap()];
    let kk = [card::parse("Ks").unwrap(), card::parse("Kd").unwrap()];

    let start = Instant::now();
    let exact = equity::exact(&[aa, kk], &[], &[]);
    let dt = start.elapsed();
    println!(
        "equity: {:>10} boards in {:>8.3?}  =>  AA {:.4} / KK {:.4}  (exact, preflop)",
        exact.trials,
        dt,
        exact.share(0),
        exact.share(1)
    );

    let n = 1_000_000;
    let start = Instant::now();
    let mc = equity::monte_carlo(&[aa, kk], &[], &[], n, 0xBEEF);
    let dt = start.elapsed();
    println!(
        "equity: {:>10} trials in {:>8.3?}  =>  AA {:.4}          (sampled, off by {:+.4})",
        n,
        dt,
        mc.share(0),
        mc.share(0) - exact.share(0)
    );

    // Ranges pay for rejection sampling and a fresh runout every trial, so this
    // is the number to watch when the solver starts asking for range equities.
    let ranges = [
        equity::Range::parse("JJ+, AKs").unwrap(),
        equity::Range::parse("22-99, AJs+, KQs").unwrap(),
    ];
    let n = 500_000;
    let start = Instant::now();
    let rr = equity::range_vs_range(&ranges, &[], &[], n, 4242);
    let dt = start.elapsed();
    println!(
        "equity: {:>10} trials in {:>8.3?}  =>  {:.4} / {:.4}      (range vs range, preflop)",
        n,
        dt,
        rr.share(0),
        rr.share(1)
    );
}

fn main() {
    bench_eval();
    bench_engine();
    bench_equity();
}
