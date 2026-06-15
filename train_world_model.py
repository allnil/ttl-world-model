import argparse

import numpy as np
import torch

from world_model import WorldModel, encode, train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="transitions.npz")
    parser.add_argument("--out", default="wm.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.data)

    states = data["states"]
    actions = data["actions"]
    next_states = data["next_states"]
    rewards = data["rewards"]
    dones = data["dones"]

    x = encode(states, actions)

    target_boards = torch.from_numpy(next_states[:, :9].astype(np.int64)) + 1
    target_rewards = torch.from_numpy(rewards.astype(np.int64)) + 1
    target_dones = torch.from_numpy(dones.astype(np.float32))

    model = WorldModel()
    train(
        model,
        x,
        target_boards,
        target_rewards,
        target_dones,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    torch.save(model.state_dict(), args.out)
    print(f"saved model checkpoint to {args.out}")


if __name__ == "__main__":
    main()
