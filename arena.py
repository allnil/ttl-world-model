import argparse
from pathlib import Path

import numpy as np
import torch

from agent import MinimaxAgent
from mcts_agent import MCTSAgent, VALUE_MODES
from ttt_env import TicTacToe
from train_policy import PolicyNet
from train_value import ValueNet
from world_model import WorldModel

AGENT_KINDS = ("random", "minimax", "minimax-value", "mcts", "puct")


class RandomAgent:
    def __init__(self, rng):
        self.rng = rng

    def choose_move(self, state):
        legal_actions = np.where(state[:9] == 0)[0]
        return int(self.rng.choice(legal_actions)), 0


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


def load_policy_model(path):
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(
            f"missing policy checkpoint: {path}. Run `uv run python train_policy.py` first."
        )
    model = PolicyNet()
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


def make_agent(
    kind,
    model,
    value_model,
    policy_model,
    depth,
    mcts_sims,
    mcts_exploration,
    mcts_value_mode,
    rng,
):
    if kind == "random":
        return RandomAgent(rng)
    if kind == "minimax":
        return MinimaxAgent(model, depth=depth, rng=rng)
    if kind == "minimax-value":
        if value_model is None:
            raise ValueError("minimax-value requires --value-checkpoint")
        return MinimaxAgent(model, depth=depth, rng=rng, value_model=value_model)
    if kind == "mcts":
        if value_model is None:
            raise ValueError("mcts requires --value-checkpoint")
        return MCTSAgent(
            model,
            value_model,
            simulations=mcts_sims,
            exploration=mcts_exploration,
            value_mode=mcts_value_mode,
            rng=rng,
        )
    if kind == "puct":
        if value_model is None:
            raise ValueError("puct requires --value-checkpoint")
        if policy_model is None:
            raise ValueError("puct requires --policy-checkpoint")
        return MCTSAgent(
            model,
            value_model,
            simulations=mcts_sims,
            exploration=mcts_exploration,
            value_mode=mcts_value_mode,
            policy_model=policy_model,
            rng=rng,
        )
    raise ValueError(f"unknown agent kind: {kind}")


def agent_kind_label(kind, depth, mcts_sims, mcts_value_mode):
    if kind == "random":
        return "random"
    if kind == "minimax":
        return f"minimax depth {depth}"
    if kind == "minimax-value":
        return f"minimax depth {depth} + value"
    if kind == "mcts":
        return f"MCTS {mcts_sims} sims ({mcts_value_mode} value)"
    if kind == "puct":
        return f"PUCT {mcts_sims} sims ({mcts_value_mode} value + policy prior)"
    raise ValueError(f"unknown agent kind: {kind}")


def play_configured_agents(
    model,
    value_model,
    policy_model,
    x_agent_kind,
    o_agent_kind,
    depth_x,
    depth_o,
    mcts_sims,
    mcts_exploration,
    mcts_value_mode,
    seed,
):
    rng = np.random.default_rng(seed)
    agent_x = make_agent(
        x_agent_kind,
        model,
        value_model,
        policy_model,
        depth_x,
        mcts_sims,
        mcts_exploration,
        mcts_value_mode,
        rng,
    )
    agent_o = make_agent(
        o_agent_kind,
        model,
        value_model,
        policy_model,
        depth_o,
        mcts_sims,
        mcts_exploration,
        mcts_value_mode,
        rng,
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


def run_configured_agents(
    model,
    value_model,
    policy_model,
    x_agent_kind,
    o_agent_kind,
    depth_x,
    depth_o,
    mcts_sims,
    mcts_exploration,
    mcts_value_mode,
    n_games,
    base_seed,
):
    return [
        play_configured_agents(
            model,
            value_model,
            policy_model,
            x_agent_kind,
            o_agent_kind,
            depth_x,
            depth_o,
            mcts_sims,
            mcts_exploration,
            mcts_value_mode,
            base_seed + game_id,
        )
        for game_id in range(n_games)
    ]


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


def result_counts(results):
    return {
        "x_wins": sum(result == 1 for result in results),
        "o_wins": sum(result == -1 for result in results),
        "draws": sum(result == 0 for result in results),
    }


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
    parser.add_argument("--policy-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--value-side",
        choices=("none", "x", "o", "both"),
        default="both",
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--depth-x", type=int, default=None)
    parser.add_argument("--depth-o", type=int, default=None)
    parser.add_argument("--x-agent", choices=AGENT_KINDS, default=None)
    parser.add_argument("--o-agent", choices=AGENT_KINDS, default=None)
    parser.add_argument("--mcts-sims", type=int, default=100)
    parser.add_argument("--mcts-exploration", type=float, default=1.4)
    parser.add_argument("--mcts-value-mode", choices=VALUE_MODES, default="expected")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--matchups", default="5:5,5:9,6:9,9:9")
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.checkpoint)
    value_model = load_value_model(args.value_checkpoint)
    policy_model = load_policy_model(args.policy_checkpoint)
    value_side = args.value_side if value_model is not None else "none"
    depth_x = args.depth if args.depth_x is None else args.depth_x
    depth_o = args.depth if args.depth_o is None else args.depth_o

    if args.x_agent is not None or args.o_agent is not None:
        if args.x_agent is None or args.o_agent is None:
            raise ValueError("--x-agent and --o-agent must be provided together")
        results = run_configured_agents(
            model,
            value_model,
            policy_model,
            args.x_agent,
            args.o_agent,
            depth_x,
            depth_o,
            args.mcts_sims,
            args.mcts_exploration,
            args.mcts_value_mode,
            args.games,
            args.seed,
        )
        print(
            f"X = {agent_kind_label(args.x_agent, depth_x, args.mcts_sims, args.mcts_value_mode)}"
        )
        print(
            f"O = {agent_kind_label(args.o_agent, depth_o, args.mcts_sims, args.mcts_value_mode)}"
        )
        summarize_results(results)
        return

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
