# poker-engine

Foundation layer for a hold'em solver: cards, hand evaluation, a seeded deck,
and a no-limit betting engine. No dependencies.

```
cargo test                          # correctness
cargo run --release --bin bench     # throughput
```

On Windows this needs the `x86_64-pc-windows-gnu` toolchain unless you have
Visual Studio installed — the default `msvc` toolchain compiles fine and then
fails at the link step, because it needs `link.exe` and the Windows SDK.

```
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu
```

## What's here

| Module | Contents |
| --- | --- |
| `card` | `u8` cards encoded `rank * 4 + suit`, parsing and formatting |
| `eval` | 5/6/7-card evaluator returning a comparable `u32` |
| `deck` | xoshiro256++ RNG, unbiased Fisher-Yates shuffle, card removal |
| `game` | NLHE state machine: legal actions, min-raise and reopening rules, side pots, showdown payouts |
| `equity` | exact enumeration, Monte Carlo, range notation, range-vs-range with card removal |

## Design decisions worth knowing

**Cards are `u8`, rank in the high bits.** `c >> 2` is the rank, `c & 3` the
suit. Both are single instructions, and a 13-bit rank mask per suit falls out
naturally, which is what the evaluator runs on.

**Evaluation returns one `u32`, larger is stronger.** Category in bits 20–23,
then five 4-bit kicker slots. Comparing two hands is one integer compare, so
the solver never pays for hand comparison in its inner loop.

**The evaluator uses no lookup tables.** It counts ranks and suits directly.
That makes it verifiable by reading it, with zero startup cost and zero memory.
It is not the fastest possible design — when profiling says evaluation is the
bottleneck, swap in a perfect-hash or two-plus-two table behind the same
`eval(&[u8]) -> u32` signature. Nothing else has to change.

There is a subtlety in there that looks like a bug and isn't: the evaluator
returns a flush immediately, before checking quads or full houses, even though
both outrank a flush. That is correct at 5–7 cards, because a flush consumes
five distinct ranks and the two remaining cards cannot make a rank appear three
or four times. The comment at that line spells out the argument. Don't
"fix" it without redoing the counting.

**The engine does not deal.** The caller supplies board cards, and the deck
takes an explicit seed. Solver work and variance reduction both depend on
replaying the identical hand many times, so hidden nondeterminism would be a
correctness bug rather than a convenience.

**Actions are "raise **to**", never "raise **by**".** Once blinds and short
all-ins are involved, "by" is ambiguous and it is the classic source of
off-by-one pot bugs.

**Chips are integers.** Never floats for money.

**Range sampling restarts the whole draw on a conflict.** This is the one thing
in `equity` that is easy to get subtly, invisibly wrong. When two players want
the same card, the sampler throws away every combo it drew that trial and
starts over. The tempting shortcut — keep what you have and redraw only the
player that clashed — biases the result: it lets a holding that blocks most of
the opponent's range turn up as often as one that blocks none of it, when in a
real deal the blocking holding is exactly the one that should appear less. The
shortcut still produces plausible numbers, which is why it survives in a lot of
code. `rejection_sampling_matches_exact_enumeration` is what catches it.

## Rules the engine actually implements

These are the ones people get wrong:

- Heads-up, the button posts the small blind and acts **first** preflop but
  **last** on every later street.
- The big blind has the option to raise when the action limps around.
- An all-in for **less than a full raise** raises the bet but does **not**
  reopen the action for players who have already acted. Getting this wrong
  inflates the game tree and yields strategies illegal at a real table.
- Folded players' chips are dead money and stay in the lowest pots they
  reached, even though those seats can't win them.
- Odd chips in a split go to the first winner left of the button.
- Once every opponent is all-in, betting is **over** — on that street and every
  street after it. The last player with chips is never offered a bet, because
  nobody is left to call one. This one is easy to get wrong in two separate
  places: the street-end check has to stop requiring that the lone player act
  (a new street clears that flag, which would reopen a dead betting round each
  time), and legal-action generation has to stop offering the raise. Chips
  still balance either way — the uncalled bet refunds through a side pot only
  the bettor is eligible for — so pot arithmetic will not catch it. What it
  costs you is a large dead subtree in the solver.

## Test coverage

52 tests. Use `cargo test --release` — three preflop enumerations dominate the
run, and they take about nine seconds unoptimised against one in release.

`equity.rs` is checked against published equity tables rather than against its
own output, which is the only way such a test says anything. AA over KK is
quoted as 81.25% to 82.64% depending on suits, and all three configurations are
pinned. The spread is the interesting part: sharing a suit with the kings
*helps* the aces, because in a shared suit they hold the nut flush card and both
players draw to a flush only one of them can win. Sampling is then tied back to
enumeration from both directions — single-combo ranges must reproduce the
specific-hand path, and a range must reproduce the average of the configurations
it contains.

Beyond the per-rule unit tests, `game.rs` runs a 2,000-hand random
walk that plays arbitrary legal actions and asserts chip conservation on every
hand — no chips created, no chips destroyed, pot always balancing against
payouts. That single invariant catches most pot-arithmetic mistakes.

It does not catch all of them, and it is worth knowing where the blind spot is.
Conservation is silent on any bug that moves chips somewhere legal-looking and
then moves them back, which is exactly what an illegal bet into an all-in table
does. Both of those had to be pinned by rule-specific tests instead. When you
add a rule here, assume conservation will not cover it.

The all-in and reopening tests are deliberately four-handed where two live
opponents are needed. Heads-up, the reopening rule and the all-opponents-all-in
rule both block the same raise, so a heads-up test passes even when one of the
two is broken.

## Next

1. **Abstraction** — action abstraction (bet-size trees) and card abstraction
   (potential-aware bucketing). `equity` is the input: bucketing wants E[HS²]
   and potential-aware metrics computed off these same enumerations.
2. **Solver** — MCCFR with external sampling over the abstracted tree.
3. **Search** — depth-limited subgame re-solving against a blueprint.
4. **Evaluation harness** — AIVAT and duplicate-hand testing, so a win rate is
   measurable in tens of thousands of hands rather than hundreds of thousands.

Before the solver, `equity` will want a caching layer — MCCFR asks for the same
holding-versus-range numbers millions of times, and at present every call
re-enumerates from scratch.
