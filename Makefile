PY := uv run python

DATA ?= transitions.npz
EXHAUSTIVE ?= transitions_exhaustive.npz
CHECKPOINT ?= wm.pt
VALUE_CHECKPOINT ?= value.pt
VALUE_SIDE ?= both
VALUE_FRACTION ?= 0.3
GAMES ?= 20000
SEED ?= 0
EPOCHS ?= 10
BATCH ?= 512
VALUE_EPOCHS ?= 200
N ?= 100
DEPTH ?= 5
MATCHUPS ?= 5:5,5:9,6:9,9:9
DEPTH_X ?= 5
DEPTH_O ?= 9
MAX_REPLAYS ?= 6
LADDER_N ?= 200
LADDER_DEPTH ?= 9
WEAK_GAMES ?= 500 1000 5000 20000
WEAK_EPOCHS ?= 30
WEAK_ARENA_N ?= 100
WEAK_DEPTH ?= 9
NAIVE_N ?= 100
NAIVE_DEPTH ?= 9

.PHONY: help sync data data-random data-exhaustive train eval value-oracle train-value eval-value arena arena-value value-generalization loss-forensics ladder weak-model weak-coverage naive-opponent notebook-check check full

help:
	@printf "Targets:\n"
	@printf "  make sync                         install/sync dependencies\n"
	@printf "  make data GAMES=20000 SEED=0      build random-game dataset\n"
	@printf "  make data-exhaustive              build exhaustive BFS dataset\n"
	@printf "  make train EPOCHS=10              train WorldModel checkpoint\n"
	@printf "  make eval                         evaluate checkpoint on exhaustive data\n"
	@printf "  make value-oracle                 build exact value labels\n"
	@printf "  make train-value                  train ValueNet checkpoint\n"
	@printf "  make eval-value                   evaluate ValueNet checkpoint\n"
	@printf "  make arena N=100 DEPTH=5          run random baseline and minimax self-play\n"
	@printf "  make arena-value                  run arena with value cutoff enabled\n"
	@printf "  make value-generalization         train value on a subset and test holdout accuracy\n"
	@printf "  make loss-forensics               replay X depth 5 losses vs O depth 9\n"
	@printf "  make ladder                       one-step/two-step/minimax vs random\n"
	@printf "  make weak-model                   weak model exact-match plus winrate\n"
	@printf "  make naive-opponent               mean-opponent planner vs random/minimax\n"
	@printf "  make notebook-check               execute clean experiments.ipynb\n"
	@printf "  make check                        compile scripts and run eval\n"
	@printf "  make full                         run the full reproducibility chain\n"
	@printf "\nUseful variables:\n"
	@printf "  DATA=$(DATA) EXHAUSTIVE=$(EXHAUSTIVE) CHECKPOINT=$(CHECKPOINT) VALUE_CHECKPOINT=$(VALUE_CHECKPOINT) VALUE_SIDE=$(VALUE_SIDE) VALUE_FRACTION=$(VALUE_FRACTION)\n"
	@printf "  GAMES=$(GAMES) SEED=$(SEED) EPOCHS=$(EPOCHS) VALUE_EPOCHS=$(VALUE_EPOCHS) BATCH=$(BATCH)\n"
	@printf "  N=$(N) DEPTH=$(DEPTH) MATCHUPS=$(MATCHUPS)\n"
	@printf "  DEPTH_X=$(DEPTH_X) DEPTH_O=$(DEPTH_O) MAX_REPLAYS=$(MAX_REPLAYS)\n"
	@printf "  LADDER_N=$(LADDER_N) LADDER_DEPTH=$(LADDER_DEPTH)\n"
	@printf "  WEAK_GAMES='$(WEAK_GAMES)' WEAK_EPOCHS=$(WEAK_EPOCHS) WEAK_ARENA_N=$(WEAK_ARENA_N) WEAK_DEPTH=$(WEAK_DEPTH)\n"
	@printf "  NAIVE_N=$(NAIVE_N) NAIVE_DEPTH=$(NAIVE_DEPTH)\n"

sync:
	uv sync

data: data-random

data-random:
	$(PY) collect_data.py --games $(GAMES) --seed $(SEED) --out $(DATA)

data-exhaustive:
	$(PY) enumerate_data.py --out $(EXHAUSTIVE)

train:
	$(PY) train_world_model.py --data $(DATA) --out $(CHECKPOINT) --epochs $(EPOCHS) --batch-size $(BATCH)

eval:
	$(PY) eval_world_model.py --checkpoint $(CHECKPOINT) --data $(EXHAUSTIVE)

value-oracle:
	$(PY) value_oracle.py

train-value:
	$(PY) train_value.py --epochs $(VALUE_EPOCHS)

eval-value:
	$(PY) eval_value.py

arena:
	$(PY) arena.py --checkpoint $(CHECKPOINT) --games $(N) --depth $(DEPTH) --seed $(SEED) --matchups "$(MATCHUPS)"

arena-value:
	$(PY) arena.py --checkpoint $(CHECKPOINT) --value-checkpoint $(VALUE_CHECKPOINT) --value-side $(VALUE_SIDE) --games $(N) --depth $(DEPTH) --seed $(SEED) --matchups "$(MATCHUPS)"

value-generalization:
	$(PY) experiment_value_generalization.py --train-fraction $(VALUE_FRACTION) --seed $(SEED)

loss-forensics:
	$(PY) experiment_loss_forensics.py --checkpoint $(CHECKPOINT) --games $(N) --seed $(SEED) --depth-x $(DEPTH_X) --depth-o $(DEPTH_O) --max-replays $(MAX_REPLAYS)

ladder:
	$(PY) experiment_ladder.py --checkpoint $(CHECKPOINT) --games $(LADDER_N) --seed $(SEED) --depth $(LADDER_DEPTH)

weak-model:
	$(PY) experiment_weak_model_coverage.py --exhaustive-data $(EXHAUSTIVE) --games $(WEAK_GAMES) --seed $(SEED) --epochs $(WEAK_EPOCHS) --arena-games $(WEAK_ARENA_N) --depth $(WEAK_DEPTH)

weak-coverage: weak-model

naive-opponent:
	$(PY) experiment_naive_opponent.py --checkpoint $(CHECKPOINT) --games $(NAIVE_N) --depth $(NAIVE_DEPTH) --seed $(SEED)

notebook-check:
	uv run jupyter nbconvert --to notebook --execute --output-dir /tmp --output ttt-world-model-experiments-check.ipynb --ExecutePreprocessor.timeout=300 experiments.ipynb
	rm -f /tmp/ttt-world-model-experiments-check.ipynb

check:
	$(PY) -m py_compile collect_data.py enumerate_data.py world_model.py train_world_model.py eval_world_model.py value_oracle.py train_value.py eval_value.py agent.py arena.py experiment_loss_forensics.py experiment_ladder.py experiment_weak_model_coverage.py experiment_value_generalization.py experiment_naive_opponent.py
	$(MAKE) eval

full: data-exhaustive data train eval arena loss-forensics ladder weak-model naive-opponent
