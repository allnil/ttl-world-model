import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def encode_value_states(states):
    boards = states[:, :9]
    players = states[:, 9:10]

    board_onehot = np.zeros((len(states), 9, 3), dtype=np.float32)
    for value, channel in [(-1, 0), (0, 1), (1, 2)]:
        board_onehot[:, :, channel] = boards == value

    x = np.concatenate(
        [board_onehot.reshape(len(states), 27), players],
        axis=1,
    )
    return torch.from_numpy(x.astype(np.float32))


class ValueNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return self.net(x)


def predict_value(model, state):
    """Return the hard value class: -1, 0, or +1.

    This is correct when the value model is used as a solved tablebase proxy
    (`value.pt` is 100% accurate). For weaker models it throws away confidence:
    51% win and 99% win both become +1.
    """
    model.eval()
    x = encode_value_states(state[None, :])

    with torch.no_grad():
        logits = model(x)

    value_class = int(logits.argmax(dim=1).item())
    return value_class - 1


def predict_value_expected(model, state):
    """Return E[value] from the model's class probabilities.

    ValueNet is trained as a 3-class classifier over {-1, 0, +1}. MCTS needs a
    scalar leaf estimate, and for an imperfect classifier the expected value is
    more informative than the hard argmax class:

        E[V] = P(+1) - P(-1)

    Draw probability contributes 0. The result is a float in [-1, +1].
    """
    model.eval()
    x = encode_value_states(state[None, :])

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    scores = torch.tensor([-1.0, 0.0, 1.0], dtype=probs.dtype)
    return float((probs * scores).sum(dim=1).item())


def train(model, x, targets, epochs=200, batch_size=512):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(x)

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        correct = 0

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]

            logits = model(x[idx])
            loss = nn.functional.cross_entropy(logits, targets[idx])

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(dim=1) == targets[idx]).sum().item()

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(
                f"epoch {epoch}: "
                f"loss {total_loss / n:.4f}, "
                f"acc {correct / n:.1%}"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("values_oracle.npz"))
    parser.add_argument("--out", type=Path, default=Path("value.pt"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.data)

    states = data["states"]
    values = data["values"]

    x = encode_value_states(states)
    targets = torch.from_numpy(values.astype(np.int64)) + 1

    model = ValueNet()
    train(model, x, targets, epochs=args.epochs, batch_size=args.batch_size)

    torch.save(model.state_dict(), args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
