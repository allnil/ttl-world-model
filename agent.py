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
                _, opponent_reward, _ = predict_next_state(
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


class MinimaxAgent:
    def __init__(self, model, depth, rng=None):
        if depth < 1:
            raise ValueError(f"minimax depth must be >= 1, got {depth}")

        self.model = model
        self.depth = int(depth)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.cache = {}
        self.model.eval()

    def cache_key(self, state, depth, root_player):
        normalized_state = np.asarray(state, dtype=np.int8).tobytes()
        return normalized_state, int(depth), int(root_player)

    def minimax_score(self, state, depth, root_player):
        key = self.cache_key(state, depth, root_player)
        if key in self.cache:
            return self.cache[key]
        actions = legal_actions_from_state(state)
        if depth <= 0 or len(actions) == 0:
            return 0
        current_player = int(state[9])
        next_boards, rewards, dones = predict_transitions(self.model, state, actions)

        scores = []
        for next_board, reward, done in zip(next_boards, rewards, dones):
            if done:
                scores.append(reward_for_player(int(reward), root_player))
            else:
                next_state = np.append(next_board, -current_player).astype(np.int8)
                scores.append(self.minimax_score(next_state, depth - 1, root_player))

        if current_player == root_player:
            result = max(scores)
        else:
            result = min(scores)

        self.cache[key] = result
        return result

    def choose_move(self, state):
        action_scores = self.score_actions(state)
        if not action_scores:
            raise ValueError("no legal actions available")

        best_score = max(score for _, score in action_scores)
        best = [action for action, score in action_scores if score == best_score]
        return int(self.rng.choice(best)), best_score

    def score_actions(self, state):
        root_player = int(state[9])
        action_scores = []

        for action in legal_actions_from_state(state):
            next_state, reward, done = predict_next_state(self.model, state, action)
            if done:
                score = reward_for_player(int(reward), root_player)
            else:
                score = self.minimax_score(next_state, self.depth - 1, root_player)
            action_scores.append((int(action), score))

        return action_scores


def minimax_score(model, state, depth, root_player):
    agent = MinimaxAgent(model, depth=max(1, int(depth)))
    return agent.minimax_score(state, depth, root_player)


def choose_move_minimax(model, state, depth, rng=None):
    agent = MinimaxAgent(model, depth=depth, rng=rng)
    return agent.choose_move(state)
