import argparse
from pathlib import Path

import numpy as np
import torch

from agent import MinimaxAgent
from arena import load_policy_model
from eval_value import load_value_model
from mcts_agent import MCTSAgent, VALUE_MODES
from world_model import WorldModel


def load_model(path):
    if not path.exists():
        raise FileNotFoundError(
            f"missing checkpoint: {path}. Run `uv run python train_world_model.py` first."
        )
    model = WorldModel()
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("wm.pt"))
    parser.add_argument("--value-checkpoint", type=Path, default=Path("value.pt"))
    parser.add_argument("--policy-checkpoint", type=Path, default=None)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exploration", type=float, default=1.4)
    parser.add_argument("--value-mode", choices=VALUE_MODES, default="expected")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    model = load_model(args.checkpoint)
    value_model = load_value_model(args.value_checkpoint)
    policy_model = load_policy_model(args.policy_checkpoint)

    state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int8)
    minimax = MinimaxAgent(model, depth=args.depth, rng=rng)
    mcts = MCTSAgent(
        model,
        value_model,
        simulations=args.simulations,
        exploration=args.exploration,
        value_mode=args.value_mode,
        policy_model=policy_model,
        rng=rng,
    )

    minimax_scores = dict(minimax.score_actions(state))
    action, q = mcts.choose_move(state)
    mcts_stats = mcts.root_action_stats()

    print("root state: empty board, X to move")
    print(f"minimax depth: {args.depth}")
    print(f"mcts simulations: {args.simulations}")
    print(f"mcts exploration: {args.exploration}")
    print(f"mcts value mode: {args.value_mode}")
    print(f"policy prior: {policy_model is not None}")
    print()
    print("action | minimax score | mcts visits | mcts q")
    for action_id, visits, value in mcts_stats:
        print(
            f"{action_id:>6} | "
            f"{minimax_scores[action_id]:>13} | "
            f"{visits:>11} | "
            f"{value:>6.3f}"
        )

    print()
    print(f"chosen action: {action}")
    print(f"chosen q: {q:.3f}")


if __name__ == "__main__":
    main()
