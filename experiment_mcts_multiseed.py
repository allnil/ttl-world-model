import argparse
from pathlib import Path

import numpy as np

from arena import (
    load_model,
    load_policy_model,
    load_value_model,
    result_counts,
    run_configured_agents,
)


MATCHUPS = {
    "mcts-random": ("mcts", "random", "MCTS X vs random O"),
    "mcts-minimax": ("mcts", "minimax", "MCTS X vs minimax O"),
    "minimax-mcts": ("minimax", "mcts", "minimax X vs MCTS O"),
    "puct-minimax": ("puct", "minimax", "PUCT X vs minimax O"),
    "minimax-puct": ("minimax", "puct", "minimax X vs PUCT O"),
}


def format_mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return f"{values.mean():.1f}+/-{values.std(ddof=0):.1f}"


def run_one(model, value_model, policy_model, matchup_key, simulations, games, seed, args):
    x_agent, o_agent, _label = MATCHUPS[matchup_key]
    results = run_configured_agents(
        model,
        value_model,
        policy_model,
        x_agent,
        o_agent,
        args.minimax_depth,
        args.minimax_depth,
        simulations,
        args.exploration,
        args.value_mode,
        games,
        seed,
    )
    return result_counts(results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("wm.pt"))
    parser.add_argument("--value-checkpoint", type=Path, default=Path("value.pt"))
    parser.add_argument("--policy-checkpoint", type=Path, default=None)
    parser.add_argument("--simulations", type=int, nargs="+", default=[10, 50, 200])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--minimax-depth", type=int, default=9)
    parser.add_argument("--exploration", type=float, default=1.4)
    parser.add_argument("--value-mode", choices=("class", "expected"), default="expected")
    parser.add_argument(
        "--matchups",
        nargs="+",
        choices=tuple(MATCHUPS),
        default=["mcts-minimax", "minimax-mcts"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.checkpoint)
    value_model = load_value_model(args.value_checkpoint)
    policy_model = load_policy_model(args.policy_checkpoint)

    print("Format: mean+/-std counts across seeds")
    print(f"games per seed: {args.games}")
    print(f"seeds: {args.seeds}")
    print(f"minimax depth: {args.minimax_depth}")
    print(f"exploration: {args.exploration}")
    print(f"value mode: {args.value_mode}")
    print()

    for matchup_key in args.matchups:
        _x_agent, _o_agent, label = MATCHUPS[matchup_key]
        print(label)
        print("sims | X wins mean+/-std | O wins mean+/-std | draws mean+/-std")

        for simulations in args.simulations:
            per_seed = [
                run_one(
                    model,
                    value_model,
                    policy_model,
                    matchup_key,
                    simulations,
                    args.games,
                    seed,
                    args,
                )
                for seed in args.seeds
            ]
            x_wins = [counts["x_wins"] for counts in per_seed]
            o_wins = [counts["o_wins"] for counts in per_seed]
            draws = [counts["draws"] for counts in per_seed]
            print(
                f"{simulations:>4} | "
                f"{format_mean_std(x_wins):>17} | "
                f"{format_mean_std(o_wins):>17} | "
                f"{format_mean_std(draws):>16}"
            )
        print()


if __name__ == "__main__":
    main()
