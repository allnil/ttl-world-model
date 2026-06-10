from collections import deque

import numpy as np

from ttt_env import TicTacToe


def board_key(board):
    return board.tobytes()


def state_with_player(board, player):
    return np.append(board, player).astype(np.int8)


def set_env(env, board, current_player):
    env.board = board.copy()
    env.current_player = current_player
    env.done = is_terminal(board, current_player)


def is_terminal(board, current_player):
    previous_player = -current_player
    if has_win(board, previous_player):
        return True
    return not np.any(board == 0)


def has_win(board, player):
    for line in TicTacToe.WIN_LINES:
        if all(board[idx] == player for idx in line):
            return True
    return False


def enumerate_transitions():
    env = TicTacToe()
    env.reset()
    start_board = env.board.copy()
    start_player = env.current_player

    queue = deque([(start_board, start_player)])
    seen_states = {(board_key(start_board), start_player)}
    transitions_by_pair = {}
    terminal_states = set()

    while queue:
        state, player = queue.popleft()
        set_env(env, state, player)
        state_id = (board_key(state), player)
        if env.done:
            terminal_states.add(state_id)
            continue

        for action in env.legal_actions():
            next_state, reward, done = env.step(int(action))
            next_player = env.current_player
            pair_id = (board_key(state), player, int(action))
            transition = (
                state_with_player(state, player),
                int(action),
                next_state.copy(),
                int(reward),
                bool(done),
            )

            existing = transitions_by_pair.get(pair_id)
            if existing is not None and not same_transition(existing, transition):
                raise AssertionError(
                    f"non-deterministic transition for action {action}"
                )
            transitions_by_pair[pair_id] = transition
            next_board = env.board.copy()
            next_state_id = (board_key(next_board), next_player)
            if next_state_id not in seen_states:
                seen_states.add(next_state_id)
                queue.append((next_board, next_player))
            set_env(env, state, player)
    return list(transitions_by_pair.values()), seen_states, terminal_states


def same_transition(left, right):
    return (
        np.array_equal(left[0], right[0])
        and left[1] == right[1]
        and np.array_equal(left[2], right[2])
        and left[3] == right[3]
        and left[4] == right[4]
    )


def save_transitions(path, transitions):
    states, actions, next_states, rewards, dones = zip(*transitions)
    np.savez(
        path,
        states=np.array(states, dtype=np.int8),
        actions=np.array(actions, dtype=np.int8),
        next_states=np.array(next_states, dtype=np.int8),
        rewards=np.array(rewards, dtype=np.int8),
        dones=np.array(dones, dtype=np.bool_),
    )


if __name__ == "__main__":
    transitions, seen_states, terminal_states = enumerate_transitions()
    save_transitions("transitions_exhaustive.npz", transitions)

    print(f"state-action transitions: {len(transitions)}")
    print(f"reachable states including terminal: {len(seen_states)}")
    print(f"terminal states: {len(terminal_states)}")
    print("saved: transitions_exhaustive.npz")
