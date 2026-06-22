## GOAL

Tinkering and experimenting with a World Model concept on the most simple example.

Educational project.

## Current architecture

This project is a tiny model-based planning stack for tic-tac-toe. It is useful
because the whole game is small enough to solve exactly, so learned components
can be checked against ground truth.

| Component | File | Contract |
| --- | --- | --- |
| Environment | `ttt_env.py` | True game rules, `state = 9 board cells + current_player` |
| Exhaustive data | `enumerate_data.py` | All reachable state-action transitions |
| World model | `world_model.py` | `state + action -> next_board, reward, done` |
| Minimax planner | `agent.py` | Exact depth-limited search inside the learned world model |
| Value oracle | `value_oracle.py` | Exact minimax value for every reachable state |
| Value model | `train_value.py` | `state -> {-1, 0, +1}` from X's perspective |
| Policy model | `train_policy.py` | `state -> action prior`, trained from oracle-optimal moves |
| MCTS / PUCT planner | `mcts_agent.py` | Budgeted tree search using `WorldModel`, `ValueNet`, and optional policy priors |
| Experiments | `arena.py`, `experiment_*.py`, `Makefile` | Reproducible command surface |

The important separation is:

```text
world model != value model != policy model != planner
```

The world model predicts transitions. The value model evaluates positions. The
policy model suggests which actions are worth looking at first. The planner
searches through imagined transitions and optionally uses value/policy models to
guide that search.

## Reproducible commands

Run these from this directory.

```bash
make eval
uv run python eval_value.py
make arena N=100 MATCHUPS=5:9,9:5
make arena-value N=100 MATCHUPS=5:9 VALUE_SIDE=x
make arena-value N=100 MATCHUPS=9:5 VALUE_SIDE=o
make value-generalization VALUE_FRACTION=0.3
make alpha-beta
make mcts MCTS_SIMS=500 DEPTH=9
make puct MCTS_SIMS=500 DEPTH=9
make arena-mcts N=50 MCTS_SIMS=100 DEPTH_X=9 DEPTH_O=9
make arena-puct N=50 MCTS_SIMS=100 DEPTH_X=9 DEPTH_O=9
make mcts-sweep N=50 MCTS_SWEEP='10 50 100 200 500' DEPTH_O=9
make mcts-sweep-weak N=50 MCTS_SWEEP='10 50 200' DEPTH_O=9
make mcts-multiseed N=30 MCTS_SWEEP='10 50 200' MCTS_SEEDS='0 1 2'
make puct-multiseed N=30 MCTS_SWEEP='10 50 200' MCTS_SEEDS='0 1 2'
make mcts-choice-analysis N=30 MCTS_SWEEP='10 50 200' MCTS_SEEDS='0 1 2'
make check
```

Useful build commands:

```bash
make data-exhaustive
make data
make train
make value-oracle
make train-value
make eval-value
make train-policy
make eval-policy
```

## Key results

World model evaluation on the exhaustive transition set:

| Examples | Board exact-match | Reward accuracy | Done accuracy |
| ---: | ---: | ---: | ---: |
| 16,167 | 100.0% | 100.0% | 100.0% |

Value model evaluation on the exact value table:

| Examples | Value accuracy | Empty-board value |
| ---: | ---: | ---: |
| 5,478 | 100.0% | 0 |

Policy model evaluation on oracle-optimal action targets:

| Examples | Top-1 optimal action accuracy | Mean probability mass on optimal actions |
| ---: | ---: | ---: |
| 4,520 | 99.8% | 92.0% |

One-sided value cutoff test, 200 games, seed 0:

| Matchup | X wins | O wins | Draws | Lesson |
| --- | ---: | ---: | ---: | --- |
| X depth 5 vs O depth 9 | 0 | 16 | 184 | Plain depth-5 X has a horizon hole |
| X depth 5 + value vs O depth 9 | 0 | 0 | 200 | Value closes the X-side horizon hole |
| X depth 9 vs O depth 5 | 146 | 0 | 54 | Plain depth-5 O is strongly exploitable |
| X depth 9 vs O depth 5 + value | 0 | 0 | 200 | Value closes the O-side horizon hole |

Generalization check, training `ValueNet` on 30% of value states and testing on
the remaining 70%:

| Split / class | Accuracy |
| --- | ---: |
| Train | 100.0% |
| Test | 79.9% |
| Test value `-1` | 72.2% |
| Test value `0` | 72.0% |
| Test value `1` | 86.8% |

Full value training is a neural tablebase: the model sees all reachable states
and compresses the solved game into weights. The 30%/70% experiment is the first
real generalization check: the model memorizes the training subset but makes
mistakes on unseen states.

Alpha-beta pruning check, empty board, depth 9:

| Search | Nodes | Cutoffs | Result |
| --- | ---: | ---: | --- |
| Plain minimax | 294,777 | 0 | Same root scores |
| Alpha-beta | 18,194 | 13,640 | Same root scores |
| Alpha-beta + value ordering | 3,829 | 2,844 | Same root scores |

Alpha-beta preserved the minimax result while reducing searched nodes by 93.8%.
Adding value-based move ordering reduced searched nodes by 98.7%. This is a
search optimization, not a new evaluator: same answer, less inference work
through the learned world model.

MCTS root search, empty board, 500 simulations, depth-9 minimax reference:

| Action | Minimax score | MCTS visits | MCTS q |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 56 | 0.355 |
| 1 | 0 | 20 | 0.050 |
| 2 | 0 | 51 | 0.336 |
| 3 | 0 | 24 | 0.089 |
| 4 | 0 | 35 | 0.233 |
| 5 | 0 | 20 | 0.050 |
| 6 | 0 | 210 | 0.657 |
| 7 | 0 | 19 | 0.008 |
| 8 | 0 | 65 | 0.383 |

This is intentionally different from minimax. MCTS is budgeted statistical
search: it spends simulations where its current tree policy sees promise. With
finite budget it can prefer a move even when exhaustive minimax says all root
moves draw under perfect play.

MCTS arena smoke test:

| Matchup | Games | X wins | O wins | Draws |
| --- | ---: | ---: | ---: | ---: |
| MCTS 100 sims as X vs minimax depth 9 as O | 20 | 0 | 0 | 20 |

MCTS budget sweep, 50 games per cell, seed 0. Counts are `X wins / O wins / draws`.
The default value mode is `expected`: ValueNet logits are converted to
`P(+1) - P(-1)`. This preserves uncertainty for imperfect value models. The old
hard-class mode is still available with `MCTS_VALUE_MODE=class`.

Perfect value (`value.pt`, 100% accurate):

| Sims | MCTS X vs random O | MCTS X vs minimax9 O | minimax9 X vs MCTS O |
| ---: | --- | --- | --- |
| 10 | 49/0/1 | 0/0/50 | 0/0/50 |
| 50 | 50/0/0 | 0/0/50 | 3/0/47 |
| 200 | 50/0/0 | 0/0/50 | 8/0/42 |

Weak value (`value_partial.pt`, trained on 30% of states, ~80% accurate):

| Sims | MCTS X vs random O | MCTS X vs minimax9 O | minimax9 X vs MCTS O |
| ---: | --- | --- | --- |
| 10 | 50/0/0 | 0/10/40 | 24/0/26 |
| 50 | 50/0/0 | 0/1/49 | 6/0/44 |
| 200 | 49/0/1 | 0/0/50 | 0/0/50 |

Two lessons. (1) **Search compensates for a weak evaluator.** With the weak value,
MCTS loses heavily to optimal minimax at 10 sims, then recovers by 200 sims. This
is the same tradeoff seen in minimax (depth-vs-value), now expressed as
sims-vs-value. (2) **MCTS is approximate.** Even with perfect value it
occasionally drops a game as O where exact minimax never would, because move
selection is stochastic at a finite budget. In a game this small exact minimax is
strictly better; MCTS only earns its keep when the tree is too large to search
exactly.

Multi-seed MCTS robustness, 30 games per seed, seeds 0/1/2. Counts are
`mean+/-std` over seeds.

Perfect value (`value.pt`, expected mode):

| Matchup | Sims | X wins | O wins | Draws |
| --- | ---: | ---: | ---: | ---: |
| MCTS X vs minimax O | 10 | 0.0+/-0.0 | 0.0+/-0.0 | 30.0+/-0.0 |
| MCTS X vs minimax O | 50 | 0.0+/-0.0 | 0.0+/-0.0 | 30.0+/-0.0 |
| MCTS X vs minimax O | 200 | 0.0+/-0.0 | 0.0+/-0.0 | 30.0+/-0.0 |
| minimax X vs MCTS O | 10 | 0.0+/-0.0 | 0.0+/-0.0 | 30.0+/-0.0 |
| minimax X vs MCTS O | 50 | 1.0+/-0.0 | 0.0+/-0.0 | 29.0+/-0.0 |
| minimax X vs MCTS O | 200 | 4.7+/-0.5 | 0.0+/-0.0 | 25.3+/-0.5 |

Weak value (`value_partial.pt`, expected mode):

| Matchup | Sims | X wins | O wins | Draws |
| --- | ---: | ---: | ---: | ---: |
| MCTS X vs minimax O | 10 | 0.0+/-0.0 | 8.0+/-0.0 | 22.0+/-0.0 |
| MCTS X vs minimax O | 50 | 0.0+/-0.0 | 1.0+/-0.0 | 29.0+/-0.0 |
| MCTS X vs minimax O | 200 | 0.0+/-0.0 | 0.0+/-0.0 | 30.0+/-0.0 |
| minimax X vs MCTS O | 10 | 13.7+/-0.5 | 0.0+/-0.0 | 16.3+/-0.5 |
| minimax X vs MCTS O | 50 | 4.3+/-0.5 | 0.0+/-0.0 | 25.7+/-0.5 |
| minimax X vs MCTS O | 200 | 0.0+/-0.0 | 0.0+/-0.0 | 30.0+/-0.0 |

The perfect-value O-side result is the important warning. More simulations do
not guarantee monotonic playing strength for this vanilla UCT implementation at
finite budget. The search backs up sampled averages and chooses by visit count,
not by an exact minimax proof. In tactical two-player games that can overvalue
risky lines until the search has enough structure or guidance to refute them.
This is a motivation for the next steps: policy priors/PUCT, exploration tuning,
and possibly minimax-style tactical backups.

MCTS choice analysis, perfect value, `minimax X vs MCTS O`, 30 games per seed,
seeds 0/1/2. This keeps the same MCTS machinery but changes the final action
recommendation rule.

| Sims | Choice rule | X wins | O wins | Draws |
| ---: | --- | ---: | ---: | ---: |
| 10 | visits | 0.0+/-0.0 | 0.0+/-0.0 | 30.0+/-0.0 |
| 10 | q | 1.0+/-0.0 | 0.0+/-0.0 | 29.0+/-0.0 |
| 10 | minimax-check top-3 | 0.0+/-0.0 | 0.0+/-0.0 | 30.0+/-0.0 |
| 50 | visits | 1.0+/-0.0 | 0.0+/-0.0 | 29.0+/-0.0 |
| 50 | q | 7.3+/-0.5 | 0.0+/-0.0 | 22.7+/-0.5 |
| 50 | minimax-check top-3 | 0.0+/-0.0 | 0.0+/-0.0 | 30.0+/-0.0 |
| 200 | visits | 4.7+/-0.5 | 0.0+/-0.0 | 25.3+/-0.5 |
| 200 | q | 5.7+/-0.5 | 0.0+/-0.0 | 24.3+/-0.5 |
| 200 | minimax-check top-3 | 0.0+/-0.0 | 0.0+/-0.0 | 30.0+/-0.0 |

Result: q-based final selection does not fix the failure; it is often worse.
But a small minimax safety check over the top-3 visited MCTS moves restores all
draws in this test. That means MCTS usually surfaces safe actions near the top,
but vanilla sampled-average/visit-count recommendation is not tactical enough
against a perfect minimax opponent.

PUCT adds the learned policy prior into selection:

```text
score(s,a) = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

`P(s,a)` is the PolicyNet prior over legal actions. It does not replace search;
it biases search toward actions that the policy believes are plausible. In this
toy project the policy is supervised from the oracle, not learned by self-play.

PUCT robustness, perfect value + policy prior, 30 games per seed, seeds 0/1/2:

| Matchup | Sims | X wins | O wins | Draws |
| --- | ---: | ---: | ---: | ---: |
| PUCT X vs minimax O | 10 | 0.0+/-0.0 | 0.0+/-0.0 | 30.0+/-0.0 |
| PUCT X vs minimax O | 50 | 0.0+/-0.0 | 0.0+/-0.0 | 30.0+/-0.0 |
| PUCT X vs minimax O | 200 | 0.0+/-0.0 | 0.0+/-0.0 | 30.0+/-0.0 |
| minimax X vs PUCT O | 10 | 0.0+/-0.0 | 0.0+/-0.0 | 30.0+/-0.0 |
| minimax X vs PUCT O | 50 | 0.0+/-0.0 | 0.0+/-0.0 | 30.0+/-0.0 |
| minimax X vs PUCT O | 200 | 0.0+/-0.0 | 0.0+/-0.0 | 30.0+/-0.0 |

Compared with vanilla UCT, PUCT fixes the O-side finite-budget failure in this
test. The reason is not magic: the policy prior makes MCTS spend its early
budget on tactically plausible moves instead of treating all legal actions as
equally worth exploring.

Reproduce:

```bash
make mcts-sweep N=50 MCTS_SWEEP='10 50 200'    # perfect value (value.pt, the default)
make mcts-sweep-weak N=50 MCTS_SWEEP='10 50 200'
make mcts-multiseed N=30 MCTS_SWEEP='10 50 200' MCTS_SEEDS='0 1 2'
make mcts-multiseed-weak N=30 MCTS_SWEEP='10 50 200' MCTS_SEEDS='0 1 2'
make mcts-choice-analysis N=30 MCTS_SWEEP='10 50 200' MCTS_SEEDS='0 1 2'
make puct-multiseed N=30 MCTS_SWEEP='10 50 200' MCTS_SEEDS='0 1 2'
```

More details live in `VALUE_HEAD_EXPERIMENT.md`.

## Progress log

- [x] Learned world model reaches 100% transition/reward/done accuracy on the
  exhaustive tic-tac-toe transition set.
- [x] Value oracle solves all 5,478 reachable states and trains a 100% accurate
  `ValueNet` on the full table.
- [x] Value cutoff closes the depth-5 horizon hole against plain depth-9
  minimax in one-sided tests.
- [x] Alpha-beta pruning preserves minimax root scores while cutting depth-9
  searched nodes from 294,777 to 18,194.
- [x] Value-based move ordering preserves root scores and cuts depth-9 searched
  nodes further to 3,829.
- [x] MCTS prototype uses `WorldModel` for expansion and `ValueNet` for leaf
  evaluation, showing budgeted tree search behavior.
- [x] MCTS is integrated into `arena.py` and has a budget sweep experiment.
- [x] MCTS leaf evaluation supports both hard value classes and expected value
  from logits; the expected mode is the default for weak value checkpoints.
- [x] Weak-value MCTS sweep shows the sims-vs-value tradeoff directly.
- [x] Multi-seed MCTS sweep reports mean/std and exposes a vanilla UCT O-side
  finite-budget failure mode.
- [x] Choice analysis shows q-based final selection does not fix the O-side
  failure, while minimax-check over top visited moves does.
- [x] PolicyNet learns oracle-optimal action priors with 99.8% top-1 optimal
  action accuracy.
- [x] PUCT integrates policy priors into MCTS selection and fixes the observed
  O-side finite-budget failure against minimax in the tested setting.

## Current status

The project now has a complete toy AlphaZero-shaped planning stack:

```text
WorldModel -> imagined transitions
ValueNet   -> leaf evaluation
PolicyNet  -> action priors
PUCT       -> policy/value-guided search
```

This is still supervised by exact tic-tac-toe oracles, not self-play RL. The
next conceptual jump is to stop using the oracle for policy targets and instead
train policy from PUCT visit counts.

## Next steps

1. **PUCT stress tests.** Vary `MCTS_EXPLORATION`, weak policy/value checkpoints,
   and tactical states to find where PUCT still fails.
2. **Self-play policy improvement.** Train policy targets from MCTS/PUCT visit
   counts instead of the exact oracle. This is the real AlphaZero-shaped loop.
3. **Batching.** Batch value/policy calls during MCTS evaluation so larger sweeps do not
   spend most of their time in one-state neural-network inference.
4. **GridWorld / MiniGrid.** Move to a slightly larger environment where exact
   enumeration may still be possible at first, then deliberately disappears.

## Bug journal (fixes and context)

Bugs that occurred during development, with their fixes and the lesson each one taught.

### 1. Player sign flip typo (`ttt_env.py`)
`self.current_player -= self.current_player` set the player to 0 instead of flipping the sign. Next move silently wrote 0 to the board, and `_check_win(0)` matched an empty line — "emptiness won" and the game ended as a fake draw.
**Fix:** `self.current_player = -self.current_player` + `assert self.current_player in (1, -1)` in `step`.
**Lesson:** turn implicit invariants into asserts; a bug crashes far from where it lives.

### 2. Trained model overwritten in the notebook
Re-running a cell that contained `model = WorldModel()` after training rebound the name to a fresh random net — eval suddenly showed 0.0%.
**Fix:** separate cells: create model / train / inference.
**Lesson:** a variable is just a name bound to an object; group notebook cells by re-run cost.

### 3. Stale kernel module + silent numpy slice
The env was upgraded to 10-dim observations, but the running kernel kept the old 9-dim version cached. `states[:, 9:10]` on a `(N, 9)` array silently returned an empty `(N, 0)` slice, so the input had 36 features instead of 37 and failed deep inside `nn.Linear` ("mat1 and mat2 shapes cannot be multiplied").
**Fix:** restart the kernel; `assert states.shape[1] == 10` at the top of `encode`.
**Lesson:** numpy out-of-range slices do not raise; validate shapes at function boundaries.

### 4. Dataset format mismatch
`transitions.npz` was collected before the observation format change (9-dim states), while `transitions_exhaustive.npz` was 10-dim.
**Fix:** re-collect the random dataset.
**Lesson:** when a data format changes, regenerate every derived artifact.

### 5. Minimax cache poisoning (`agent.py`)
Cache key `(board, player)` omitted `root_player` and `depth`. Scores computed from X's perspective could be reused for O with the wrong sign, and horizon-truncated values mixed with full evaluations.
**Fix:** key = `(board, player, depth, root_player)`. (Cleaner alternative: canonicalize all scores to X's perspective.)
**Lesson:** a cache key must include everything the cached value depends on.

### 6. Double forward pass
A refactor leftover called `model(x)` once outside `torch.no_grad()` (building an autograd graph) and again inside, discarding the first result.
**Fix:** delete the stray call.

### 7. Contract mismatch after the batch refactor
`predict_next_state` passed a scalar action into the new batch API; `np.asarray(5)` is a 0-d array → `TypeError: len() of unsized object`.
**Fix:** pass `[int(action)]` and unpack `[0]` from each returned array.
**Lesson:** when a function's contract changes, update every caller.

### 8. Assert before normalization
`assert actions.ndim == 1` ran before `np.asarray`, so a plain Python list crashed the assert itself (`AttributeError: 'list' object has no attribute 'ndim'`).
**Fix:** normalize input first, validate second.

### 9. Frankenstein function
A tie-break snippet pasted after the old selection loop referenced undefined `all_actions` / `all_scores` / `rng` → `NameError`.
**Fix:** restructure `choose_move_minimax`: collect scores, then tie-break; pass `rng` as a parameter.
**Lesson:** don't merge two generations of code; rewrite the function as a whole.

### 10. Misleading traceback
After heavy edits under autoreload, traceback arrows pointed at comments and import lines ("Could not get source..."). Kernel code objects and file text had diverged — not a real code bug.
**Lesson:** if a traceback points at comments, trust the exception type, read bottom-up, restart the kernel after refactors.

### 11. Deterministic "100 games"
The agent-vs-agent arena had no randomness left (the `seed` parameter was unused): all 100 games were the same game replayed.
**Fix:** random tie-breaking among equally-scored moves, rng passed in from the arena seed.
**Lesson:** a dead parameter is a smell; statistics require variation.
