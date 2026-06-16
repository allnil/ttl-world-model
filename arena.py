import argparse
from pathlib import Path

import numpy as np
import torch

from agent import MinimaxAgent
from ttt_env import TicTacToe
from train_value import ValueNet
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


def load_value_model(path):
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(
            f"missing value checkpoint: {path}. Run `uv run python train_value.py` first."
        )
    model = ValueNet()
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def value_for_side(value_model, value_side, player):
    if value_model is None or value_side == "none":
        return None
    if value_side == "both":
        return value_model
    if value_side == "x" and player == 1:
        return value_model
    if value_side == "o" and player == -1:
        return value_model
    return None


def agent_label(depth, player, value_model, value_side):
    has_value = value_for_side(value_model, value_side, player) is not None
    suffix = " + value" if has_value else ""
    return f"minimax depth {depth}{suffix}"


def play_agent_vs_random(model, depth, seed, value_model=None, value_side="none"):
    rng = np.random.default_rng(seed)
    agent = MinimaxAgent(
        model,
        depth=depth,
        rng=rng,
        value_model=value_for_side(value_model, value_side, player=1),
    )
    env = TicTacToe()
    state = env.reset()
    done = False
    while not done:
        player = int(state[9])
        if player == 1:
            action, _score = agent.choose_move(state)
        else:
            action = int(rng.choice(env.legal_actions()))
        state, reward, done = env.step(action)
    return int(reward)


def play_agent_vs_agent(
    model,
    depth_x,
    depth_o,
    seed,
    value_model=None,
    value_side="none",
):
    rng = np.random.default_rng(seed)
    agent_x = MinimaxAgent(
        model,
        depth=depth_x,
        rng=rng,
        value_model=value_for_side(value_model, value_side, player=1),
    )
    agent_o = MinimaxAgent(
        model,
        depth=depth_o,
        rng=rng,
        value_model=value_for_side(value_model, value_side, player=-1),
    )
    env = TicTacToe()
    state = env.reset()
    done = False

    while not done:
        player = int(state[9])

        if player == 1:
            action, _score = agent_x.choose_move(state)
        else:
            action, _score = agent_o.choose_move(state)

        state, reward, done = env.step(action)

    return int(reward)


def summarize_results(results):
    x_wins = sum(result == 1 for result in results)
    o_wins = sum(result == -1 for result in results)
    draws = sum(result == 0 for result in results)
    print(f"games: {len(results)}")
    print(f"X wins: {x_wins}")
    print(f"O wins: {o_wins}")
    print(f"draws: {draws}")


def run_agent_vs_random(
    model,
    depth,
    n_games,
    base_seed,
    value_model=None,
    value_side="none",
):
    return [
        play_agent_vs_random(
            model,
            depth=depth,
            seed=base_seed + i,
            value_model=value_model,
            value_side=value_side,
        )
        for i in range(n_games)
    ]


def run_agent_vs_agent(
    model,
    depth_x,
    depth_o,
    n_games,
    base_seed,
    value_model=None,
    value_side="none",
):
    return [
        play_agent_vs_agent(
            model,
            depth_x=depth_x,
            depth_o=depth_o,
            seed=base_seed + i,
            value_model=value_model,
            value_side=value_side,
        )
        for i in range(n_games)
    ]


def parse_matchups(value):
    matchups = []
    for item in value.split(","):
        left, right = item.split(":")
        matchups.append((int(left), int(right)))
    return matchups


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("wm.pt"))
    parser.add_argument("--value-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--value-side",
        choices=("none", "x", "o", "both"),
        default="both",
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--matchups", default="5:5,5:9,6:9,9:9")
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.checkpoint)
    value_model = load_value_model(args.value_checkpoint)
    value_side = args.value_side if value_model is not None else "none"

    random_results = run_agent_vs_random(
        model,
        depth=args.depth,
        n_games=args.games,
        base_seed=args.seed,
        value_model=value_model,
        value_side=value_side,
    )
    print(f"X = {agent_label(args.depth, 1, value_model, value_side)}")
    print("O = random")
    summarize_results(random_results)

    for depth_x, depth_o in parse_matchups(args.matchups):
        print()
        self_play_results = run_agent_vs_agent(
            model,
            depth_x=depth_x,
            depth_o=depth_o,
            n_games=args.games,
            base_seed=args.seed,
            value_model=value_model,
            value_side=value_side,
        )
        print(f"X = {agent_label(depth_x, 1, value_model, value_side)}")
        print(f"O = {agent_label(depth_o, -1, value_model, value_side)}")
        summarize_results(self_play_results)


if __name__ == "__main__":
    main()
