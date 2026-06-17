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
| MCTS planner | `mcts_agent.py` | Budgeted tree search using `WorldModel` expansion and `ValueNet` leaf evaluation |
| Experiments | `arena.py`, `experiment_*.py`, `Makefile` | Reproducible command surface |

The important separation is:

```text
world model != value model != planner
```

The world model predicts transitions. The value model evaluates positions. The
planner searches through imagined transitions and optionally calls the value
model at the depth cutoff.

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
make arena-mcts N=50 MCTS_SIMS=100 DEPTH_X=9 DEPTH_O=9
make mcts-sweep N=50 MCTS_SWEEP='10 50 100 200 500' DEPTH_O=9
make mcts-sweep-weak N=50 MCTS_SWEEP='10 50 200' DEPTH_O=9
make mcts-multiseed N=30 MCTS_SWEEP='10 50 200' MCTS_SEEDS='0 1 2'
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

Reproduce:

```bash
make mcts-sweep N=50 MCTS_SWEEP='10 50 200'    # perfect value (value.pt, the default)
make mcts-sweep-weak N=50 MCTS_SWEEP='10 50 200'
make mcts-multiseed N=30 MCTS_SWEEP='10 50 200' MCTS_SEEDS='0 1 2'
make mcts-multiseed-weak N=30 MCTS_SWEEP='10 50 200' MCTS_SEEDS='0 1 2'
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

## Next steps

1. **Policy head.** Add `policy(state) -> action prior` so search does not need
   to treat all legal moves equally. Value reduces effective depth; policy
   reduces effective breadth.
2. **PUCT.** Add policy priors to MCTS selection and compare MCTS vs PUCT at the
   same simulation budgets.
3. **UCT failure analysis.** Compare visit-count move choice against q-based
   choice, tune exploration, and test minimax/solver-style backups on tactical
   states where vanilla MCTS as O loses.
4. **Batching.** Batch value calls during MCTS evaluation so larger sweeps do not
   spend most of their time in one-state neural-network inference.
5. **GridWorld / MiniGrid.** Move to a slightly larger environment where exact
   enumeration may still be possible at first, then deliberately disappears.

# bugs which occured during development with fixes and motivation/context around:

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
