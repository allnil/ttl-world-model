import argparse

import numpy as np

from ttt_env import TicTacToe


def play_random_game(env, rng):
    transitions = []
    state = env.reset()
    done = False
    while not done:
        action = rng.choice(env.legal_actions())
        next_state, reward, done = env.step(action)
        transitions.append((state, action, next_state, reward, done))
        state = next_state
    return transitions


def collect_dataset(n_games, seed=0):
    rng = np.random.default_rng(seed)
    env = TicTacToe()
    all_transitions = []
    for _ in range(n_games):
        all_transitions.extend(play_random_game(env, rng))
    return all_transitions


def dataset_unique_transitions(transitions):
    tss_set = set()
    for state, action, *_ in transitions:
        tss_set.add((state.tobytes(), action))
    return len(tss_set)


def save_dataset(path, transitions):
    states, actions, next_states, rewards, dones = zip(*transitions)
    np.savez(
        path,
        states=np.array(states, dtype=np.int8),
        actions=np.array(actions, dtype=np.int8),
        next_states=np.array(next_states, dtype=np.int8),
        rewards=np.array(rewards, dtype=np.int8),
        dones=np.array(dones, dtype=np.bool_),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="transitions.npz")
    return parser.parse_args()


def main():
    args = parse_args()
    data = collect_dataset(args.games, seed=args.seed)
    save_dataset(args.out, data)
    print(f"games: {args.games}")
    print(f"seed: {args.seed}")
    print(f"collected transitions: {len(data)}")
    print(f"unique state-action pairs: {dataset_unique_transitions(data)}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
