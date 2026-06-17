import argparse
from pathlib import Path

from arena import (
    load_model,
    load_value_model,
    result_counts,
    run_configured_agents,
)


def format_counts(counts):
    return f"{counts['x_wins']}/{counts['o_wins']}/{counts['draws']}"


def run_matchup(
    model,
    value_model,
    x_agent,
    o_agent,
    depth_x,
    depth_o,
    simulations,
    games,
    seed,
    exploration,
    value_mode,
):
    results = run_configured_agents(
        model,
        value_model,
        x_agent,
        o_agent,
        depth_x,
        depth_o,
        simulations,
        exploration,
        value_mode,
        games,
        seed,
    )
    return result_counts(results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("wm.pt"))
    parser.add_argument("--value-checkpoint", type=Path, default=Path("value.pt"))
    parser.add_argument("--simulations", type=int, nargs="+", default=[10, 50, 100, 200, 500])
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimax-depth", type=int, default=9)
    parser.add_argument("--exploration", type=float, default=1.4)
    parser.add_argument("--value-mode", choices=("class", "expected"), default="expected")
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.checkpoint)
    value_model = load_value_model(args.value_checkpoint)

    print("Format: X wins / O wins / draws")
    print(f"games per matchup: {args.games}")
    print(f"minimax depth: {args.minimax_depth}")
    print(f"exploration: {args.exploration}")
    print(f"value mode: {args.value_mode}")
    print()
    print("sims | MCTS X vs random O | MCTS X vs minimax9 O | minimax9 X vs MCTS O")

    for simulations in args.simulations:
        mcts_vs_random = run_matchup(
            model,
            value_model,
            "mcts",
            "random",
            args.minimax_depth,
            args.minimax_depth,
            simulations,
            args.games,
            args.seed,
            args.exploration,
            args.value_mode,
        )
        mcts_vs_minimax = run_matchup(
            model,
            value_model,
            "mcts",
            "minimax",
            args.minimax_depth,
            args.minimax_depth,
            simulations,
            args.games,
            args.seed,
            args.exploration,
            args.value_mode,
        )
        minimax_vs_mcts = run_matchup(
            model,
            value_model,
            "minimax",
            "mcts",
            args.minimax_depth,
            args.minimax_depth,
            simulations,
            args.games,
            args.seed,
            args.exploration,
            args.value_mode,
        )
        print(
            f"{simulations:>4} | "
            f"{format_counts(mcts_vs_random):>18} | "
            f"{format_counts(mcts_vs_minimax):>22} | "
            f"{format_counts(minimax_vs_mcts):>22}"
        )


if __name__ == "__main__":
    main()
