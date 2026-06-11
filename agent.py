import numpy as np
import torch

from world_model import encode


def predict_transitions(model, state, actions):
    actions = np.asarray(actions, dtype=np.int64)

    assert actions.ndim == 1, f"expected array of actions, get {actions!r}"

    states = np.repeat(state[None, :], len(actions), axis=0)
    x = encode(states, actions)
    with torch.no_grad():
        board_logits, reward_logits, done_logits = model(x)
    next_boards = (board_logits.argmax(dim=2) - 1).numpy().astype(np.int8)  # (N, 9)
    rewards = (reward_logits.argmax(dim=1) - 1).numpy()  # (N,)
    dones = (torch.sigmoid(done_logits) > 0.5).numpy()  # (N,)
    return next_boards, rewards, dones


def predict_next_state(model, state, action):
    next_boards, rewards, dones = predict_transitions(model, state, [int(action)])
    next_player = -int(state[9])
    next_state = np.append(next_boards[0], next_player).astype(np.int8)
    return next_state, int(rewards[0]), bool(dones[0])


def legal_actions_from_state(state):
    assert state.shape == (10,), f"expected state shape (10,), got {state.shape}"
    board = state[:9]
    return np.where(board == 0)[0]


def choose_move_one_step(model, state):
    model.eval()
    player = int(state[9])
    best_action = None
    best_score = -10

    for action in legal_actions_from_state(state):
        _, pred_reward, _ = predict_next_state(model, state, action)

        score = pred_reward if player == 1 else -pred_reward

        if score > best_score:
            best_score = score
            best_action = action

    if best_action is None:
        raise ValueError("no legal actions available")

    return int(best_action), best_score


def reward_for_player(reward, player):
    return reward if player == 1 else -reward


def choose_move_two_step(model, state):
    model.eval()
    player = int(state[9])
    best_action = None
    best_score = -10

    for action in legal_actions_from_state(state):
        next_state, reward, done = predict_next_state(model, state, action)

        if done:
            score = reward_for_player(reward, player)
        else:
            opponent_scores = []

            for opponent_action in legal_actions_from_state(next_state):
                _, opponent_reward, opponent_done = predict_next_state(
                    model, next_state, opponent_action
                )
                opponent_scores.append(reward_for_player(opponent_reward, player))

            score = min(opponent_scores) if opponent_scores else 0

        if score > best_score:
            best_score = score
            best_action = action

    if best_action is None:
        raise ValueError("no legal actions available")

    return int(best_action), best_score


cache = {}


def minimax_score(model, state, depth, root_player):
    key = (state[:9].tobytes(), int(state[9]), depth, root_player)
    if key in cache:
        return cache[key]

    actions = legal_actions_from_state(state)

    if depth == 0 or len(actions) == 0:
        return 0

    current_player = int(state[9])
    next_boards, rewards, dones = predict_transitions(model, state, actions)

    scores = []
    for next_board, reward, done in zip(next_boards, rewards, dones):
        if done:
            scores.append(reward_for_player(int(reward), root_player))
        else:
            next_state = np.append(next_board, -current_player).astype(np.int8)
            scores.append(minimax_score(model, next_state, depth - 1, root_player))
    result = 0
    if current_player == root_player:
        result = max(scores)
    else:
        result = min(scores)

    cache[key] = result
    return result


def choose_move_minimax(model, state, depth, rng):
    model.eval()
    root_player = int(state[9])
    actions = legal_actions_from_state(state)
    if len(actions) == 0:
        raise ValueError("no legal actions available")

    scores = []
    for action in actions:
        next_state, reward, done = predict_next_state(model, state, action)
        if done:
            scores.append(reward_for_player(int(reward), root_player))
        else:
            scores.append(minimax_score(model, next_state, depth - 1, root_player))

    best_score = max(scores)
    best = [int(a) for a, s in zip(actions, scores) if s == best_score]
    return int(rng.choice(best)), best_score
