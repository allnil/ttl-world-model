import argparse
from pathlib import Path

import numpy as np

from agent import MinimaxAgent, choose_move_one_step, choose_move_two_step
from arena import load_model
from ttt_env import TicTacToe


def choose_action(model, planner, state, minimax_agent):
    if planner == "one-step":
        action, _score = choose_move_one_step(model, state)
        return action
    if planner == "two-step":
        action, _score = choose_move_two_step(model, state)
        return action
    if planner == "minimax":
        action, _score = minimax_agent.choose_move(state)
        return action
    raise ValueError(f"unknown planner: {planner}")


def play_planner_vs_random(model, planner, role, seed, depth):
    rng = np.random.default_rng(seed)
    minimax_agent = MinimaxAgent(model, depth=depth, rng=rng)
    env = TicTacToe()
    state = env.reset()
    done = False

    while not done:
        player = int(state[9])
        planner_turn = (role == "X" and player == 1) or (role == "O" and player == -1)

        if planner_turn:
            action = choose_action(model, planner, state, minimax_agent)
        else:
            action = int(rng.choice(env.legal_actions()))

        state, reward, done = env.step(action)

    return int(reward)


def summarize_for_role(results, role):
    x_wins = sum(result == 1 for result in results)
    o_wins = sum(result == -1 for result in results)
    draws = sum(result == 0 for result in results)
    agent_wins = x_wins if role == "X" else o_wins
    agent_losses = o_wins if role == "X" else x_wins
    return x_wins, o_wins, draws, agent_wins, agent_losses


def run_ladder(model, planners, roles, games, seed, depth):
    print("planner,role,games,x_wins,o_wins,draws,agent_wins,agent_losses")
    for planner in planners:
        for role in roles:
            results = [
                play_planner_vs_random(
                    model,
                    planner=planner,
                    role=role,
                    seed=seed + game_idx,
                    depth=depth,
                )
                for game_idx in range(games)
            ]
            x_wins, o_wins, draws, agent_wins, agent_losses = summarize_for_role(
                results, role
            )
            print(
                f"{planner},{role},{games},{x_wins},{o_wins},{draws},"
                f"{agent_wins},{agent_losses}"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("wm.pt"))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument(
        "--planners",
        nargs="+",
        default=["one-step", "two-step", "minimax"],
        choices=["one-step", "two-step", "minimax"],
    )
    parser.add_argument("--roles", nargs="+", default=["X", "O"], choices=["X", "O"])
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.checkpoint)
    run_ladder(
        model,
        planners=args.planners,
        roles=args.roles,
        games=args.games,
        seed=args.seed,
        depth=args.depth,
    )


if __name__ == "__main__":
    main()
