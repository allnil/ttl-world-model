# Value Head Experiment

Date: 2026-06-16

This experiment tests whether a learned value cutoff fixes the minimax horizon
hole. The value model is trained from the exact oracle table produced by
`value_oracle.py`, so this is a neural tablebase experiment, not yet a
generalization experiment.

## Pipeline

| Step | Artifact | Meaning |
| --- | --- | --- |
| Oracle | `values_oracle.npz` | Exact minimax value for all 5,478 reachable states |
| Training | `value.pt` | `ValueNet(28 -> 128 -> 128 -> 3)` trained on oracle labels |
| Evaluation | `eval_value.py` | Confirms `value.pt` reloads and predicts all oracle labels |
| Planning | `MinimaxAgent(..., value_model=...)` | Replaces the depth-cutoff `return 0` with `ValueNet(state)` |

## Value Accuracy

Command:

```bash
uv run python eval_value.py
```

Result:

| Examples | Accuracy | Empty-board value |
| ---: | ---: | ---: |
| 5,478 | 100.0% | 0 |

## Isolated Horizon Test

Baseline command:

```bash
make arena N=200 MATCHUPS=5:9,9:5 SEED=0
```

Value-side commands:

```bash
make arena-value N=200 MATCHUPS=5:9 VALUE_SIDE=x SEED=0
make arena-value N=200 MATCHUPS=9:5 VALUE_SIDE=o SEED=0
```

Results:

| Matchup | X wins | O wins | Draws | Interpretation |
| --- | ---: | ---: | ---: | --- |
| X depth 5 vs O depth 9 | 0 | 16 | 184 | Plain depth-5 X still has a horizon hole |
| X depth 5 + value vs O depth 9 | 0 | 0 | 200 | Value cutoff closes the X-side horizon hole |
| X depth 9 vs O depth 5 | 146 | 0 | 54 | Plain depth-5 O is strongly exploitable |
| X depth 9 vs O depth 5 + value | 0 | 0 | 200 | Value cutoff closes the O-side horizon hole |

The key test is one-sided: only the weaker depth-5 player receives the value
cutoff, while the depth-9 opponent remains plain minimax. This isolates the
effect of value from the easier "both players have value" comparison.

## Lesson

The value head reduces effective search depth by replacing "unknown means draw"
at the search boundary with a position evaluation. In this toy game the value
labels come from a complete oracle, so the model is compressing a solved game
into neural weights. The next experiment is to remove that luxury: train on only
part of the value table and measure generalization on held-out states.

## Generalization Dessert

Command:

```bash
make value-generalization VALUE_FRACTION=0.3 SEED=0
```

Split:

| Split | Examples | Class counts `{-1, 0, 1}` |
| --- | ---: | --- |
| Train | 1,643 | `{-1: 435, 0: 317, 1: 891}` |
| Test | 3,835 | `{-1: 1039, 0: 751, 1: 2045}` |

Result:

| Metric | Accuracy |
| --- | ---: |
| Train | 100.0% |
| Test | 79.9% |
| Test value `-1` | 72.2% |
| Test value `0` | 72.0% |
| Test value `1` | 86.8% |

This is the first non-tablebase value experiment. The model can memorize the
30% training subset perfectly, but held-out value prediction is weaker,
especially for O-win and draw states. That is the real AlphaZero/MuZero-shaped
problem: useful value functions must generalize beyond exact solved labels.
