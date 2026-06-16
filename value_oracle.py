from enumerate_data import has_win
import numpy as np


def solve_value(board, player, cache):
    key = (board.tobytes(), player)
    if key in cache:
        return cache[key]
    if has_win(board, 1):
        cache[key] = 1
        return 1
    if has_win(board, -1):
        cache[key] = -1
        return -1
    empties = np.where(board == 0)[0]
    if len(empties) == 0:
        cache[key] = 0
        return 0
    child_values = [
        solve_value(_place(board, a, player), -player, cache) for a in empties
    ]
    v = max(child_values) if player == 1 else min(child_values)
    cache[key] = v
    return v


def _place(board, action, player):
    next_board = board.copy()
    action = int(action)
    assert next_board[action] == 0, f"cell {action} is occupied"
    next_board[action] = int(player)
    return next_board


def main():
    cache = {}
    start_board = np.zeros(9, dtype=np.int8)

    value = solve_value(start_board, 1, cache)

    print("empty-board value:", value)
    print("states in cache:", len(cache))

    states = []
    values = []
    for (board_bytes, player), value in cache.items():
        board = np.frombuffer(board_bytes, dtype=np.int8).copy()
        state = np.append(board, player).astype(np.int8)
        states.append(state)
        values.append(value)
    np.savez(
        "values_oracle.npz",
        states=np.array(states, dtype=np.int8),
        values=np.array(values, dtype=np.int8),
    )
    print("saved: values_oracle.npz")


if __name__ == "__main__":
    main()
