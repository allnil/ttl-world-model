import argparse
from pathlib import Path

import numpy as np

from agent import MinimaxAgent, predict_next_state, reward_for_player
from arena import load_model, load_value_model, result_counts
from mcts_agent import MCTSAgent, VALUE_MODES
from ttt_env import TicTacToe


CHOICE_MODES = ("visits", "q", "minimax-check")


def choose_with_minimax_check(model, root, root_player, depth, top_k, rng):
    """Choose among the most-visited MCTS actions using exact minimax scores.

    This is a diagnostic hybrid, not a pure MCTS rule. It asks:
    "Did MCTS put the correct move somewhere near the top, but choose the wrong
    final action by visits/q?" If this fixes losses, final action selection was
    the problem. If it does not, the tree policy failed to surface safe actions.
    """
    ranked = sorted(
        root.children.items(),
        key=lambda item: item[1].visits,
        reverse=True,
    )
    candidates = ranked[:top_k]
    minimax = MinimaxAgent(model, depth=depth, rng=rng)

    scored = []
    for action, child in candidates:
        if child.terminal_value is not None:
            score = child.terminal_value
        else:
            score = minimax.minimax_score(child.state, depth - 1, root_player)
        scored.append((int(action), score))

    best_score = max(score for _action, score in scored)
    best_actions = [action for action, score in scored if score == best_score]
    return int(rng.choice(best_actions))


def mcts_choose_action(agent, state, choice_mode, minimax_depth, top_k, rng):
    action, _q = agent.choose_move(state)
    root = agent.last_root
    root_player = int(state[9])

    if choice_mode == "visits":
        return action
    if choice_mode == "q":
        return agent.choose_from_root(root, mode="q")
    if choice_mode == "minimax-check":
        return choose_with_minimax_check(
            agent.model,
            root,
            root_player,
            minimax_depth,
            top_k,
            rng,
        )
    raise ValueError(f"unknown choice mode: {choice_mode}")


def play_game(
    model,
    value_model,
    mcts_side,
    choice_mode,
    simulations,
    minimax_depth,
    top_k,
    exploration,
    value_mode,
    seed,
):
    rng = np.random.default_rng(seed)
    env = TicTacToe()
    state = env.reset()
    done = False

    mcts = MCTSAgent(
        model,
        value_model,
        simulations=simulations,
        exploration=exploration,
        value_mode=value_mode,
        rng=rng,
    )
    minimax = MinimaxAgent(model, depth=minimax_depth, rng=rng)

    while not done:
        player = int(state[9])
        if player == mcts_side:
            action = mcts_choose_action(
                mcts,
                state,
                choice_mode,
                minimax_depth,
                top_k,
                rng,
            )
        else:
            action, _score = minimax.choose_move(state)
        state, reward, done = env.step(action)

    return int(reward)


def run_games(
    model,
    value_model,
    mcts_side,
    choice_mode,
    simulations,
    games,
    seed,
    args,
):
    return [
        play_game(
            model,
            value_model,
            mcts_side,
            choice_mode,
            simulations,
            args.minimax_depth,
            args.top_k,
            args.exploration,
            args.value_mode,
            seed + game_id,
        )
        for game_id in range(games)
    ]


def format_mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return f"{values.mean():.1f}+/-{values.std(ddof=0):.1f}"


def summarize_across_seeds(model, value_model, mcts_side, choice_mode, simulations, args):
    per_seed = [
        result_counts(
            run_games(
                model,
                value_model,
                mcts_side,
                choice_mode,
                simulations,
                args.games,
                seed,
                args,
            )
        )
        for seed in args.seeds
    ]
    return {
        "x_wins": format_mean_std([counts["x_wins"] for counts in per_seed]),
        "o_wins": format_mean_std([counts["o_wins"] for counts in per_seed]),
        "draws": format_mean_std([counts["draws"] for counts in per_seed]),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("wm.pt"))
    parser.add_argument("--value-checkpoint", type=Path, default=Path("value.pt"))
    parser.add_argument("--simulations", type=int, nargs="+", default=[10, 50, 200])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--mcts-side", choices=("x", "o"), default="o")
    parser.add_argument("--choice-modes", nargs="+", choices=CHOICE_MODES, default=list(CHOICE_MODES))
    parser.add_argument("--minimax-depth", type=int, default=9)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--exploration", type=float, default=1.4)
    parser.add_argument("--value-mode", choices=VALUE_MODES, default="expected")
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.checkpoint)
    value_model = load_value_model(args.value_checkpoint)
    mcts_side = 1 if args.mcts_side == "x" else -1
    matchup = "MCTS X vs minimax O" if mcts_side == 1 else "minimax X vs MCTS O"

    print("Format: mean+/-std counts across seeds")
    print(f"matchup: {matchup}")
    print(f"games per seed: {args.games}")
    print(f"seeds: {args.seeds}")
    print(f"minimax depth: {args.minimax_depth}")
    print(f"top-k for minimax-check: {args.top_k}")
    print(f"exploration: {args.exploration}")
    print(f"value mode: {args.value_mode}")
    print()
    print("sims | choice        | X wins mean+/-std | O wins mean+/-std | draws mean+/-std")

    for simulations in args.simulations:
        for choice_mode in args.choice_modes:
            counts = summarize_across_seeds(
                model,
                value_model,
                mcts_side,
                choice_mode,
                simulations,
                args,
            )
            print(
                f"{simulations:>4} | "
                f"{choice_mode:<13} | "
                f"{counts['x_wins']:>17} | "
                f"{counts['o_wins']:>17} | "
                f"{counts['draws']:>16}"
            )


if __name__ == "__main__":
    main()
