import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from agent import (
    legal_actions_from_state,
    predict_next_state,
    predict_transitions,
    reward_for_player,
)
from train_value import ValueNet, predict_value
from world_model import WorldModel


"""
Teaching note: what this file is for.

This is an experiment, not the production planner used by `arena.py`.
The goal is to compare two searches that should return the same scores:

1. Plain minimax:
   - recursively visits every child up to `depth`;
   - returns max score when the root player is to move;
   - returns min score when the opponent is to move.

2. Alpha-beta minimax:
   - computes the same minimax scores;
   - carries two bounds, `alpha` and `beta`, while searching;
   - stops exploring a branch when that branch can no longer change the
     decision above it.

How to read the output:

`score` must match between minimax and alpha-beta for every root action.
If scores differ, alpha-beta is wrong. `nodes` should be much smaller for
alpha-beta. `cutoffs` counts how often alpha-beta skipped the rest of a branch.

How to touch this file safely:

- If you are learning alpha-beta mechanics, edit only `alpha_beta_score`.
- If you want to test deeper searches, change `ALPHA_BETA_DEPTH` in the
  Makefile or run `make alpha-beta ALPHA_BETA_DEPTH=9`.
- Value-based move ordering is optional. It changes only the order in which
  children are searched, not the score calculation. If ordering changes a score,
  there is a bug.
- Keep the assertion in `main()`. The whole point is that alpha-beta is an
  optimization of minimax, not a different decision rule.
"""


@dataclass
class SearchStats:
    nodes: int = 0
    cutoffs: int = 0


def load_model(path):
    if not path.exists():
        raise FileNotFoundError(
            f"missing checkpoint: {path}. Run `uv run python train_world_model.py` first."
        )
    model = WorldModel()
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_value_model(path):
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(
            f"missing value checkpoint: {path}. Run `uv run python train_value.py` first."
        )
    model = ValueNet()
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def ordered_children(actions, next_boards, rewards, dones, current_player, root_player, value_model):
    children = []
    for action, next_board, reward, done in zip(actions, next_boards, rewards, dones):
        children.append((0, int(action), next_board, int(reward), bool(done)))

    if value_model is None:
        return children

    ordered = []
    for _order_score, action, next_board, reward, done in children:
        if done:
            order_score = reward_for_player(reward, root_player)
        else:
            next_state = np.append(next_board, -current_player).astype(np.int8)
            value_x = predict_value(value_model, next_state)
            order_score = reward_for_player(value_x, root_player)
        ordered.append((order_score, action, next_board, reward, done))

    reverse = current_player == root_player
    ordered.sort(key=lambda item: item[0], reverse=reverse)
    return ordered


def minimax_score(model, state, depth, root_player, stats):
    stats.nodes += 1
    actions = legal_actions_from_state(state)
    if len(actions) == 0 or depth <= 0:
        return 0

    current_player = int(state[9])
    next_boards, rewards, dones = predict_transitions(model, state, actions)
    child_scores = []

    for next_board, reward, done in zip(next_boards, rewards, dones):
        if done:
            score = reward_for_player(int(reward), root_player)
        else:
            next_state = np.append(next_board, -current_player).astype(np.int8)
            score = minimax_score(model, next_state, depth - 1, root_player, stats)
        child_scores.append(score)

    return max(child_scores) if current_player == root_player else min(child_scores)


def alpha_beta_score(
    model,
    state,
    depth,
    root_player,
    alpha,
    beta,
    stats,
    value_model=None,
):
    stats.nodes += 1
    actions = legal_actions_from_state(state)
    if len(actions) == 0 or depth <= 0:
        return 0

    current_player = int(state[9])
    next_boards, rewards, dones = predict_transitions(model, state, actions)
    children = ordered_children(
        actions,
        next_boards,
        rewards,
        dones,
        current_player,
        root_player,
        value_model,
    )

    if current_player == root_player:
        best = -float("inf")
        for _order_score, _action, next_board, reward, done in children:
            if done:
                score = reward_for_player(int(reward), root_player)
            else:
                next_state = np.append(next_board, -current_player).astype(np.int8)
                score = alpha_beta_score(
                    model,
                    next_state,
                    depth - 1,
                    root_player,
                    alpha,
                    beta,
                    stats,
                    value_model,
                )
            best = max(best, score)

            # alpha is the best score the maximizing side can already force.
            # Once alpha reaches beta, the minimizing parent has a better
            # option elsewhere and will never choose this branch.
            alpha = max(alpha, best)
            if alpha >= beta:
                stats.cutoffs += 1
                break
        return best

    best = float("inf")
    for _order_score, _action, next_board, reward, done in children:
        if done:
            score = reward_for_player(int(reward), root_player)
        else:
            next_state = np.append(next_board, -current_player).astype(np.int8)
            score = alpha_beta_score(
                model,
                next_state,
                depth - 1,
                root_player,
                alpha,
                beta,
                stats,
                value_model,
            )
        best = min(best, score)

        # beta is the best score the minimizing side can already force.
        # Once beta falls to alpha, the maximizing parent has a better option
        # elsewhere and will never choose this branch.
        beta = min(beta, best)
        if alpha >= beta:
            stats.cutoffs += 1
            break
    return best


def score_root_actions(model, state, depth, search_fn, value_model=None):
    root_player = int(state[9])
    rows = []
    total = SearchStats()

    for action in legal_actions_from_state(state):
        stats = SearchStats()
        next_state, reward, done = predict_next_state(model, state, action)
        if done:
            score = reward_for_player(reward, root_player)
        elif search_fn == "minimax":
            score = minimax_score(model, next_state, depth - 1, root_player, stats)
        elif search_fn == "alpha-beta":
            score = alpha_beta_score(
                model,
                next_state,
                depth - 1,
                root_player,
                -float("inf"),
                float("inf"),
                stats,
                value_model,
            )
        else:
            raise ValueError(f"unknown search_fn: {search_fn}")

        total.nodes += stats.nodes
        total.cutoffs += stats.cutoffs
        rows.append((int(action), int(score), stats.nodes, stats.cutoffs))

    return rows, total


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("wm.pt"))
    parser.add_argument("--value-checkpoint", type=Path, default=Path("value.pt"))
    parser.add_argument("--ordering", choices=("none", "value"), default="none")
    parser.add_argument("--depth", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.checkpoint)
    value_model = load_value_model(args.value_checkpoint) if args.ordering == "value" else None
    state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int8)

    minimax_rows, minimax_total = score_root_actions(
        model,
        state,
        depth=args.depth,
        search_fn="minimax",
    )
    alpha_beta_rows, alpha_beta_total = score_root_actions(
        model,
        state,
        depth=args.depth,
        search_fn="alpha-beta",
        value_model=value_model,
    )

    print(f"root state: empty board, X to move")
    print(f"depth: {args.depth}")
    print(f"alpha-beta ordering: {args.ordering}")
    print()
    print("action | minimax score/nodes | alpha-beta score/nodes/cutoffs")
    for minimax_row, alpha_beta_row in zip(minimax_rows, alpha_beta_rows):
        action, minimax_score_value, minimax_nodes, _ = minimax_row
        alpha_action, alpha_score_value, alpha_nodes, alpha_cutoffs = alpha_beta_row
        if action != alpha_action or minimax_score_value != alpha_score_value:
            raise AssertionError(
                f"search mismatch: minimax={minimax_row}, alpha_beta={alpha_beta_row}"
            )
        print(
            f"{action:>6} | "
            f"{minimax_score_value:>5} / {minimax_nodes:<7} | "
            f"{alpha_score_value:>5} / {alpha_nodes:<7} / {alpha_cutoffs}"
        )

    reduction = 1.0 - (alpha_beta_total.nodes / minimax_total.nodes)
    print()
    print(f"minimax nodes: {minimax_total.nodes}")
    print(f"alpha-beta nodes: {alpha_beta_total.nodes}")
    print(f"alpha-beta cutoffs: {alpha_beta_total.cutoffs}")
    print(f"node reduction: {reduction:.1%}")


if __name__ == "__main__":
    main()
