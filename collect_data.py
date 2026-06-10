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


if __name__ == "__main__":
    dataset_sizes = [10, 100, 500, 1000, 5000, 10000, 100000]
    for size in dataset_sizes:
        data = collect_dataset(size)
        states, actions, next_states, rewards, dones = zip(*data)
        np.savez(
            "transitions.npz",
            states=np.array(states, dtype=np.int8),
            actions=np.array(actions, dtype=np.int8),
            next_states=np.array(next_states, dtype=np.int8),
            rewards=np.array(rewards, dtype=np.int8),
            dones=np.array(dones),
        )
        print(f"collected transitions: {len(data)}")
        cnt = dataset_unique_transitions(data)
        print(f"number of unit transitions state-action: {cnt}")
