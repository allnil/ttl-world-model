import argparse
from pathlib import Path

import numpy as np
import torch

from world_model import WorldModel, encode


def load_model(path):
    if not path.exists():
        raise FileNotFoundError(
            f"missing pytorch checkpoint: {path}. Run `uv run python train_world_model.py` first."
        )

    model = WorldModel()
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def make_targets(data):
    target_boards = torch.from_numpy(data["next_states"][:, :9].astype(np.int64)) + 1
    target_rewards = torch.from_numpy(data["rewards"].astype(np.int64)) + 1
    target_dones = torch.from_numpy(data["dones"].astype(np.float32))
    return target_boards, target_rewards, target_dones


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("wm.pt"))
    parser.add_argument("--data", type=Path, default=Path("transitions_exhaustive.npz"))
    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.data)
    model = load_model(args.checkpoint)
    x = encode(data["states"], data["actions"])
    target_boards, target_rewards, target_dones = make_targets(data)
    with torch.no_grad():
        board_logits, reward_logits, done_logits = model(x)
    predicted_boards = board_logits.argmax(dim=2)
    predicted_rewards = reward_logits.argmax(dim=1)
    predicted_dones = torch.sigmoid(done_logits) > 0.5
    board_exact = (predicted_boards == target_boards).all(dim=1).float().mean()
    board_cell = (predicted_boards == target_boards).float().mean()
    reward_acc = (predicted_rewards == target_rewards).float().mean()
    done_acc = (predicted_dones == target_dones.bool()).float().mean()
    print(f"examples: {len(x)}")
    print(f"board exact-match: {board_exact:.1%}")
    print(f"board cell accuracy: {board_cell:.1%}")
    print(f"reward accuracy: {reward_acc:.1%}")
    print(f"done accuracy: {done_acc:.1%}")


if __name__ == "__main__":
    main()
