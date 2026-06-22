import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from enumerate_data import has_win
from value_oracle import _place
from train_value import encode_value_states


class PolicyNet(nn.Module):
    """State -> action logits.

    ValueNet answers "how good is this state?" PolicyNet answers "which actions
    should search look at first?" The output has 9 logits, one per board cell.
    Illegal moves are masked at inference/evaluation time instead of being
    removed from the network output; this keeps the interface fixed-size.
    """

    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 9),
        )

    def forward(self, x):
        return self.net(x)


def build_policy_targets(states, values):
    """Build soft optimal-action targets from the exact value oracle.

    `values_oracle.npz` stores V*(state) from X's perspective:
      +1 means X can force a win, -1 means O can force a win, 0 is draw.

    A policy target is a distribution over legal optimal moves. If several
    moves are equally optimal, each gets equal probability. This is better than
    arbitrarily picking one move, because tic-tac-toe has many symmetric ties.
    """
    value_by_key = {
        (state[:9].astype(np.int8).tobytes(), int(state[9])): int(value)
        for state, value in zip(states, values)
    }

    kept_states = []
    targets = []
    optimal_action_sets = []

    for state in states:
        board = state[:9].astype(np.int8)
        player = int(state[9])
        if has_win(board, 1) or has_win(board, -1):
            continue
        legal_actions = np.where(board == 0)[0]
        if len(legal_actions) == 0:
            continue

        child_values = []
        for action in legal_actions:
            child_board = _place(board, int(action), player)
            child_key = (child_board.tobytes(), -player)
            child_values.append(value_by_key[child_key])

        # One-ply minimax through the value oracle: X maximizes the resulting
        # child value, O minimizes it. The optimal moves are the ones that reach
        # this best value (there are often several, by symmetry).
        best_value = max(child_values) if player == 1 else min(child_values)
        optimal_actions = [
            int(action)
            for action, child_value in zip(legal_actions, child_values)
            if child_value == best_value
        ]

        target = np.zeros(9, dtype=np.float32)
        target[optimal_actions] = 1.0 / len(optimal_actions)

        kept_states.append(state)
        targets.append(target)
        optimal_action_sets.append(optimal_actions)

    return (
        np.array(kept_states, dtype=np.int8),
        np.array(targets, dtype=np.float32),
        optimal_action_sets,
    )


def policy_loss(logits, target_probs):
    # Soft-target cross-entropy: -sum(target * log p). The target is a
    # distribution over the optimal moves (not a single hard label), so this is
    # the same loss AlphaZero uses to match its network policy to the MCTS visit
    # distribution. Equals KL(target || p) up to a target-only constant.
    log_probs = nn.functional.log_softmax(logits, dim=1)
    return -(target_probs * log_probs).sum(dim=1).mean()


def train(model, x, targets, epochs=200, batch_size=512):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(x)

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            logits = model(x[idx])
            loss = policy_loss(logits, targets[idx])

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(idx)

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch}: loss {total_loss / n:.4f}")


def masked_policy_probs(model, state, legal_actions):
    """Return normalized policy probabilities over legal actions only."""
    model.eval()
    x = encode_value_states(state[None, :])
    with torch.no_grad():
        logits = model(x)[0]

    # Mask illegal moves by adding -1e9 to their logits before softmax, so they
    # receive ~0 probability. The network keeps a fixed 9-cell output; legality
    # is applied here at inference rather than baked into the architecture.
    mask = torch.full_like(logits, -1e9)
    legal_actions = [int(action) for action in legal_actions]
    mask[legal_actions] = 0.0
    probs = torch.softmax(logits + mask, dim=0).cpu().numpy()
    return {action: float(probs[action]) for action in legal_actions}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("values_oracle.npz"))
    parser.add_argument("--out", type=Path, default=Path("policy.pt"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    data = np.load(args.data)
    states = data["states"]
    values = data["values"]
    policy_states, targets, _optimal_action_sets = build_policy_targets(states, values)

    print(f"policy examples: {len(policy_states)}")
    print(f"excluded terminal states: {len(states) - len(policy_states)}")

    x = encode_value_states(policy_states)
    y = torch.from_numpy(targets)

    model = PolicyNet()
    train(model, x, y, epochs=args.epochs, batch_size=args.batch_size)

    torch.save(model.state_dict(), args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
