//! Core primitives for a hold'em solver: cards, a hand evaluator, a deck,
//! and a no-limit betting engine.
//!
//! Everything here is deliberately allocation-free on the hot paths and
//! deterministic — the deck takes an explicit seed and the betting engine
//! never deals for you. Solver work needs to replay the exact same hand
//! millions of times, so hidden nondeterminism is a bug, not a convenience.

pub mod card;
pub mod deck;
pub mod equity;
pub mod eval;
pub mod game;
