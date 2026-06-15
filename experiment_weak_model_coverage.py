import argparse

import numpy as np
import torch

from arena import play_agent_vs_random
from collect_data import collect_dataset
from world_model import WorldModel, encode, train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exhaustive-data", default="transitions_exhaustive.npz")
    parser.add_argument("--games", nargs="+", type=int, default=[500, 1000, 5000, 20000])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--arena-games", type=int, default=100)
    parser.add_argument("--depth", type=int, default=9)
    return parser.parse_args()


def main():
    args = parse_args()
    e = np.load(args.exhaustive_data)
    xe = encode(e["states"], e["actions"])
    target_boards = torch.from_numpy(e["next_states"][:, :9].astype(np.int64)) + 1
    target_rewards = torch.from_numpy(e["rewards"].astype(np.int64)) + 1
    target_dones = torch.from_numpy(e["dones"].astype(np.float32))

    print(
        "train_games,board_exact,reward_acc,done_acc,"
        "arena_games,depth,x_wins,o_wins,draws"
    )

    for n_games in args.games:
        data = collect_dataset(n_games, seed=args.seed)
        s, a, ns, r, d = (np.array(v) for v in zip(*data))
        x = encode(s, a)
        tb = torch.from_numpy(ns[:, :9].astype(np.int64)) + 1
        tr = torch.from_numpy(r.astype(np.int64)) + 1
        td = torch.from_numpy(d.astype(np.float32))
        m = WorldModel()
        train(m, x, tb, tr, td, epochs=args.epochs)
        with torch.no_grad():
            board_logits, reward_logits, done_logits = m(xe)

        board_exact = (
            (board_logits.argmax(dim=2) == target_boards).all(dim=1).float().mean()
        )
        reward_acc = (reward_logits.argmax(dim=1) == target_rewards).float().mean()
        done_acc = (
            (torch.sigmoid(done_logits) > 0.5) == target_dones.bool()
        ).float().mean()

        arena_results = [
            play_agent_vs_random(m, depth=args.depth, seed=args.seed + game_idx)
            for game_idx in range(args.arena_games)
        ]
        x_wins = sum(result == 1 for result in arena_results)
        o_wins = sum(result == -1 for result in arena_results)
        draws = sum(result == 0 for result in arena_results)

        print(
            f"{n_games},{board_exact:.1%},{reward_acc:.1%},{done_acc:.1%},"
            f"{args.arena_games},{args.depth},{x_wins},{o_wins},{draws}"
        )


if __name__ == "__main__":
    main()
