import argparse
from pathlib import Path

import numpy as np
import torch

from train_value import ValueNet, encode_value_states, train


def class_counts(values):
    counts = dict(zip(*np.unique(values, return_counts=True)))
    return {value: int(counts.get(value, 0)) for value in (-1, 0, 1)}


def accuracy(model, x, targets):
    model.eval()
    with torch.no_grad():
        logits = model(x)
    predicted = logits.argmax(dim=1)
    return (predicted == targets).float().mean().item(), predicted


def print_per_class_accuracy(predicted, targets):
    for value, class_id in [(-1, 0), (0, 1), (1, 2)]:
        mask = targets == class_id
        if mask.any():
            acc = (predicted[mask] == targets[mask]).float().mean().item()
            print(f"test acc value {value}: {acc:.1%} ({int(mask.sum())} examples)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("values_oracle.npz"))
    parser.add_argument("--train-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("value_partial.pt"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = np.load(args.data)
    states = data["states"]
    values = data["values"]

    n = len(states)
    train_n = int(round(n * args.train_fraction))
    perm = rng.permutation(n)
    train_idx = perm[:train_n]
    test_idx = perm[train_n:]

    x_train = encode_value_states(states[train_idx])
    y_train = torch.from_numpy(values[train_idx].astype(np.int64)) + 1
    x_test = encode_value_states(states[test_idx])
    y_test = torch.from_numpy(values[test_idx].astype(np.int64)) + 1

    print(f"total examples: {n}")
    print(f"train examples: {len(train_idx)} ({args.train_fraction:.0%})")
    print(f"test examples: {len(test_idx)}")
    print(f"train class counts: {class_counts(values[train_idx])}")
    print(f"test class counts: {class_counts(values[test_idx])}")

    model = ValueNet()
    train(
        model,
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    train_acc, _ = accuracy(model, x_train, y_train)
    test_acc, test_predicted = accuracy(model, x_test, y_test)
    print(f"final train accuracy: {train_acc:.1%}")
    print(f"final test accuracy: {test_acc:.1%}")
    print_per_class_accuracy(test_predicted, y_test)

    torch.save(model.state_dict(), args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
