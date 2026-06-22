import math
from dataclasses import dataclass, field

import numpy as np

from agent import (
    legal_actions_from_state,
    predict_next_state,
    reward_for_player,
)
from train_value import predict_value, predict_value_expected
from train_policy import masked_policy_probs

VALUE_MODES = ("class", "expected")
CHOICE_MODES = ("visits", "q")


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

If a policy model is provided, this becomes a small PUCT search:

    score(s,a) = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

P(s,a) is the policy prior. It does not replace search; it biases search toward
actions the policy thinks are plausible. With no policy model, the agent falls
back to vanilla UCT.
"""


@dataclass
class MCTSNode:
    state: np.ndarray
    parent: "MCTSNode | None" = None
    action: int | None = None
    untried_actions: list[int] = field(default_factory=list)
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    action_priors: dict[int, float] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
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
        policy_model=None,
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
        self.policy_model = policy_model
        self.rng = rng if rng is not None else np.random.default_rng()
        self.model.eval()
        self.value_model.eval()
        if self.policy_model is not None:
            self.policy_model.eval()
        self.last_root = None

    def make_node(self, state, parent=None, action=None, prior=0.0, terminal_value=None):
        actions = (
            []
            if terminal_value is not None
            else list(legal_actions_from_state(state))
        )
        action_priors = self.action_priors(state, actions)
        if self.policy_model is None:
            self.rng.shuffle(actions)
        else:
            # `expand` pops from the end, so ascending sort expands high-prior
            # actions first. PUCT will continue using the same priors later.
            actions = sorted(actions, key=lambda a: action_priors[int(a)])
        return MCTSNode(
            state=np.asarray(state, dtype=np.int8),
            parent=parent,
            action=action,
            untried_actions=[int(action) for action in actions],
            action_priors=action_priors,
            prior=float(prior),
            terminal_value=terminal_value,
        )

    def action_priors(self, state, actions):
        actions = [int(action) for action in actions]
        if not actions:
            return {}
        if self.policy_model is None:
            uniform = 1.0 / len(actions)
            return {action: uniform for action in actions}
        return masked_policy_probs(self.policy_model, state, actions)

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

        action = self.choose_from_root(root, mode="visits")
        return action, root.children[action].q

    def choose_from_root(self, root, mode="visits"):
        """Choose a final action from an already-built MCTS root.

        This does not run more simulations. It is only the final recommendation
        rule. `visits` is the robust-child rule used by AlphaZero-style MCTS;
        `q` is useful for debugging whether the tree statistics already contain
        a better action that visit-count selection ignores.
        """
        if mode not in CHOICE_MODES:
            raise ValueError(f"mode must be one of {CHOICE_MODES}, got {mode!r}")
        if not root.children:
            raise ValueError("no legal actions available")

        if mode == "visits":
            best_score = max(child.visits for child in root.children.values())
            best_actions = [
                action
                for action, child in root.children.items()
                if child.visits == best_score
            ]
        else:
            best_score = max(child.q for child in root.children.values())
            best_actions = [
                action
                for action, child in root.children.items()
                if child.q == best_score
            ]

        return int(self.rng.choice(best_actions))

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
            prior=node.action_priors.get(int(action), 0.0),
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
            if self.policy_model is None:
                exploration = self.exploration * math.sqrt(
                    math.log(node.visits + 1) / child.visits
                )
            else:
                # PUCT: policy prior P(s,a) scales exploration. Good priors make
                # promising/legal tactical moves get attention earlier.
                exploration = (
                    self.exploration
                    * child.prior
                    * math.sqrt(max(1, node.visits))
                    / (1 + child.visits)
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
