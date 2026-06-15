## GOAL

Tinkering and experimenting with a World Model concept on the most simple example.

Educational project.

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
