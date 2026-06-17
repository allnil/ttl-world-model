import math
from dataclasses import dataclass, field

import numpy as np

from agent import (
    legal_actions_from_state,
    predict_next_state,
    reward_for_player,
)
from train_value import predict_value, predict_value_expected

VALUE_MODES = ("class", "expected")


"""
Teaching note: MCTS in this project.

Minimax and alpha-beta ask: "what is the exact minimax answer up to depth d?"
MCTS asks a different question: "given a fixed simulation budget, which parts of
the tree look worth spending more compute on?"

One MCTS simulation has four phases:

1. Selection:
   Start at the root and repeatedly choose a child using a UCB-style score.
   This balances exploitation (high average value) and exploration (low visit
   count).

2. Expansion:
   When we reach a node with untried legal actions, add one child to the tree.
   The child state is produced by the learned WorldModel.

3. Evaluation:
   Estimate the new leaf. Here we use ValueNet instead of random rollouts.
   There are two useful modes:

   - class:    argmax(logits) -> {-1, 0, +1}
   - expected: softmax(logits) -> P(+1) - P(-1)

   `class` is fine for the perfect value checkpoint. `expected` is the better
   default for weak checkpoints because it preserves uncertainty.

4. Backup:
   Push the leaf value back through the selected path by updating visit counts
   and accumulated values.

Important invariant:

All node values in this file are stored from the root player's perspective.
That means X-rooted searches treat +1 as good for X, while O-rooted searches
flip signs through `reward_for_player`.

This is not AlphaZero yet. AlphaZero-style MCTS also uses a policy prior P(s,a)
inside the selection formula. This file intentionally starts with UCB only, so
the next policy-head step has a clean place to attach.
"""


@dataclass
class MCTSNode:
    state: np.ndarray
    parent: "MCTSNode | None" = None
    action: int | None = None
    untried_actions: list[int] = field(default_factory=list)
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    terminal_value: int | None = None

    @property
    def q(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    @property
    def is_terminal(self):
        return self.terminal_value is not None


class MCTSAgent:
    def __init__(
        self,
        model,
        value_model,
        simulations=100,
        exploration=1.4,
        value_mode="expected",
        rng=None,
    ):
        if simulations < 1:
            raise ValueError(f"simulations must be >= 1, got {simulations}")
        if value_mode not in VALUE_MODES:
            raise ValueError(f"value_mode must be one of {VALUE_MODES}, got {value_mode!r}")

        self.model = model
        self.value_model = value_model
        self.simulations = int(simulations)
        self.exploration = float(exploration)
        self.value_mode = value_mode
        self.rng = rng if rng is not None else np.random.default_rng()
        self.model.eval()
        self.value_model.eval()
        self.last_root = None

    def make_node(self, state, parent=None, action=None, terminal_value=None):
        actions = (
            []
            if terminal_value is not None
            else list(legal_actions_from_state(state))
        )
        self.rng.shuffle(actions)
        return MCTSNode(
            state=np.asarray(state, dtype=np.int8),
            parent=parent,
            action=action,
            untried_actions=[int(action) for action in actions],
            terminal_value=terminal_value,
        )

    def choose_move(self, state):
        root_player = int(state[9])
        root = self.make_node(state)

        for _ in range(self.simulations):
            leaf = self.select_and_expand(root, root_player)
            value = self.evaluate_leaf(leaf, root_player)
            self.backup(leaf, value)

        self.last_root = root
        if not root.children:
            raise ValueError("no legal actions available")

        best_visits = max(child.visits for child in root.children.values())
        best_actions = [
            action
            for action, child in root.children.items()
            if child.visits == best_visits
        ]
        action = int(self.rng.choice(best_actions))
        return action, root.children[action].q

    def select_and_expand(self, node, root_player):
        while not node.is_terminal:
            if node.untried_actions:
                return self.expand(node, root_player)
            if not node.children:
                return node
            node = self.select_child(node, root_player)
        return node

    def expand(self, node, root_player):
        action = node.untried_actions.pop()
        next_state, reward, done = predict_next_state(self.model, node.state, action)
        terminal_value = reward_for_player(reward, root_player) if done else None
        child = self.make_node(
            next_state,
            parent=node,
            action=action,
            terminal_value=terminal_value,
        )
        node.children[action] = child
        return child

    def select_child(self, node, root_player):
        current_player = int(node.state[9])
        maximize_for_root = current_player == root_player

        def ucb(child):
            # `child.q` is from root perspective. At opponent nodes, lower q is
            # better for the opponent, so use -q as exploitation.
            exploitation = child.q if maximize_for_root else -child.q
            exploration = self.exploration * math.sqrt(
                math.log(node.visits + 1) / child.visits
            )
            return exploitation + exploration

        return max(node.children.values(), key=ucb)

    def evaluate_leaf(self, node, root_player):
        if node.terminal_value is not None:
            return node.terminal_value

        if self.value_mode == "class":
            value_x = predict_value(self.value_model, node.state)
        else:
            value_x = predict_value_expected(self.value_model, node.state)
        return reward_for_player(value_x, root_player)

    def backup(self, node, value):
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent

    def root_action_stats(self):
        if self.last_root is None:
            return []
        return sorted(
            [
                (action, child.visits, child.q)
                for action, child in self.last_root.children.items()
            ],
            key=lambda item: item[0],
        )
