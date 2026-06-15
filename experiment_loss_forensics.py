import argparse
from pathlib import Path

import numpy as np

from agent import MinimaxAgent
from arena import load_model, play_agent_vs_agent
from ttt_env import TicTacToe

SYMBOLS = {1: "X", -1: "O", 0: "."}


def format_board(board):
    rows = []
    for row in board.reshape(3, 3):
        rows.append(" ".join(SYMBOLS[int(value)] for value in row))
    return "\n".join(rows)


def format_scores(action_scores, chosen_action):
    cells = []
    for action, score in action_scores:
        marker = "*" if action == chosen_action else ""
        cells.append(f"{action}:{score}{marker}")
    return " ".join(cells)


def choose_from_scores(agent, action_scores):
    best_score = max(score for _, score in action_scores)
    best_actions = [action for action, score in action_scores if score == best_score]
    return int(agent.rng.choice(best_actions)), best_score


def find_loss_seeds(model, depth_x, depth_o, games, seed):
    loss_seeds = []
    for offset in range(games):
        game_seed = seed + offset
        reward = play_agent_vs_agent(model, depth_x, depth_o, game_seed)
        if reward == -1:
            loss_seeds.append(game_seed)
    return loss_seeds


def replay_with_candidate_scores(model, depth_x, depth_o, seed):
    rng = np.random.default_rng(seed)
    agent_x = MinimaxAgent(model, depth=depth_x, rng=rng)
    agent_o = MinimaxAgent(model, depth=depth_o, rng=rng)

    env = TicTacToe()
    state = env.reset()
    done = False
    move_num = 0

    print(f"===== loss replay seed={seed} =====")
    while not done:
        move_num += 1
        player = int(state[9])
        agent = agent_x if player == 1 else agent_o
        action_scores = agent.score_actions(state)
        action, score = choose_from_scores(agent, action_scores)

        print()
        print(format_board(state[:9]))
        print(
            f"move {move_num}: {SYMBOLS[player]} depth={agent.depth} "
            f"chosen={action} score={score}"
        )
        print(f"candidate scores: {format_scores(action_scores, action)}")

        state, reward, done = env.step(action)

    print()
    print(format_board(state[:9]))
    print(f"final reward: {reward}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("wm.pt"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth-x", type=int, default=5)
    parser.add_argument("--depth-o", type=int, default=9)
    parser.add_argument("--max-replays", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.checkpoint)
    loss_seeds = find_loss_seeds(
        model,
        depth_x=args.depth_x,
        depth_o=args.depth_o,
        games=args.games,
        seed=args.seed,
    )

    print(f"X depth={args.depth_x} vs O depth={args.depth_o}")
    print(f"games: {args.games}")
    print(f"loss seeds for X: {loss_seeds}")
    print(f"loss count: {len(loss_seeds)}")

    for loss_seed in loss_seeds[: args.max_replays]:
        print()
        replay_with_candidate_scores(
            model,
            depth_x=args.depth_x,
            depth_o=args.depth_o,
            seed=loss_seed,
        )


if __name__ == "__main__":
    main()
