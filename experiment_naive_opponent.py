import argparse
from pathlib import Path

import numpy as np

from agent import (
    MinimaxAgent,
    legal_actions_from_state,
    predict_next_state,
    predict_transitions,
    reward_for_player,
)
from arena import load_model
from ttt_env import TicTacToe


class RandomAgent:
    def __init__(self, rng):
        self.rng = rng

    def choose_move(self, state):
        actions = legal_actions_from_state(state)
        if len(actions) == 0:
            raise ValueError("no legal actions available")
        return int(self.rng.choice(actions)), None


class NaiveOpponentAgent:
    def __init__(self, model, depth, rng=None):
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.model = model
        self.depth = int(depth)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.cache = {}
        self.model.eval()

    def cache_key(self, state, depth, root_player):
        normalized_state = np.asarray(state, dtype=np.int8).tobytes()
        return normalized_state, int(depth), int(root_player)

    def score_state(self, state, depth, root_player):
        key = self.cache_key(state, depth, root_player)
        if key in self.cache:
            return self.cache[key]

        actions = legal_actions_from_state(state)
        if depth <= 0 or len(actions) == 0:
            return 0.0

        current_player = int(state[9])
        next_boards, rewards, dones = predict_transitions(self.model, state, actions)

        child_scores = []
        for next_board, reward, done in zip(next_boards, rewards, dones):
            if done:
                score = float(reward_for_player(int(reward), root_player))
            else:
                next_state = np.append(next_board, -current_player).astype(np.int8)
                score = self.score_state(next_state, depth - 1, root_player)
            child_scores.append(score)

        if current_player == root_player:
            result = max(child_scores)
        else:
            result = float(np.mean(child_scores))

        self.cache[key] = result
        return result

    def score_actions(self, state):
        root_player = int(state[9])
        action_scores = []

        for action in legal_actions_from_state(state):
            next_state, reward, done = predict_next_state(self.model, state, action)
            if done:
                score = float(reward_for_player(int(reward), root_player))
            else:
                score = self.score_state(next_state, self.depth - 1, root_player)
            action_scores.append((int(action), score))

        return action_scores

    def choose_move(self, state):
        action_scores = self.score_actions(state)
        if not action_scores:
            raise ValueError("no legal actions available")
        best_score = max(score for _, score in action_scores)
        best = [action for action, score in action_scores if score == best_score]
        return int(self.rng.choice(best)), best_score


def make_agent(kind, model, depth, rng):
    if kind == "random":
        return RandomAgent(rng)
    if kind == "minimax":
        return MinimaxAgent(model, depth=depth, rng=rng)
    if kind == "naive":
        return NaiveOpponentAgent(model, depth=depth, rng=rng)
    raise ValueError(f"unknown agent kind: {kind}")


def play_match(model, x_kind, o_kind, depth, seed):
    rng = np.random.default_rng(seed)
    x_agent = make_agent(x_kind, model, depth, rng)
    o_agent = make_agent(o_kind, model, depth, rng)

    env = TicTacToe()
    state = env.reset()
    done = False

    while not done:
        player = int(state[9])
        agent = x_agent if player == 1 else o_agent
        action, _score = agent.choose_move(state)
        state, reward, done = env.step(action)

    return int(reward)


def summarize(results):
    x_wins = sum(result == 1 for result in results)
    o_wins = sum(result == -1 for result in results)
    draws = sum(result == 0 for result in results)
    return x_wins, o_wins, draws


def run_scenario(model, name, x_kind, o_kind, games, depth, seed):
    results = [
        play_match(
            model,
            x_kind=x_kind,
            o_kind=o_kind,
            depth=depth,
            seed=seed + game_idx,
        )
        for game_idx in range(games)
    ]
    x_wins, o_wins, draws = summarize(results)
    print(f"{name},{x_kind},{o_kind},{games},{depth},{x_wins},{o_wins},{draws}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("wm.pt"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.checkpoint)

    print("scenario,x_agent,o_agent,games,depth,x_wins,o_wins,draws")
    run_scenario(
        model,
        "naive_vs_random",
        x_kind="naive",
        o_kind="random",
        games=args.games,
        depth=args.depth,
        seed=args.seed,
    )
    run_scenario(
        model,
        "minimax_vs_random",
        x_kind="minimax",
        o_kind="random",
        games=args.games,
        depth=args.depth,
        seed=args.seed,
    )
    run_scenario(
        model,
        "random_vs_naive",
        x_kind="random",
        o_kind="naive",
        games=args.games,
        depth=args.depth,
        seed=args.seed,
    )
    run_scenario(
        model,
        "random_vs_minimax",
        x_kind="random",
        o_kind="minimax",
        games=args.games,
        depth=args.depth,
        seed=args.seed,
    )
    run_scenario(
        model,
        "naive_vs_minimax",
        x_kind="naive",
        o_kind="minimax",
        games=args.games,
        depth=args.depth,
        seed=args.seed,
    )
    run_scenario(
        model,
        "minimax_vs_naive",
        x_kind="minimax",
        o_kind="naive",
        games=args.games,
        depth=args.depth,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
