import numpy as np
import torch

import torch.nn as nn


def encode(states, actions):
    assert states.shape[1] == 10, f"expected 10, got {states.shape}"
    boards = states[:, :9]
    players = states[:, 9:10]
    board_onehot = np.zeros((len(states), 9, 3), dtype=np.float32)
    for value, channel in [(-1, 0), (0, 1), (1, 2)]:
        board_onehot[:, :, channel] = boards == value
    action_onehot = np.zeros((len(states), 9), dtype=np.float32)
    action_onehot[np.arange(len(states)), actions] = 1.0
    x = np.concatenate(
        [board_onehot.reshape(len(states), 27), players, action_onehot], axis=1
    )
    return torch.from_numpy(x.astype(np.float32))


class WorldModel(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(37, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.board_head = nn.Linear(hidden, 27)
        self.reward_head = nn.Linear(hidden, 3)
        self.done_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return (
            self.board_head(h).view(-1, 9, 3),
            self.reward_head(h),
            self.done_head(h).squeeze(-1),
        )


def compute_loss(model, x, target_boards, target_rewards, target_dones):
    board_logits, reward_logits, done_logits = model(x)
    ce = nn.functional.cross_entropy
    board_loss = ce(board_logits.reshape(-1, 3), target_boards.reshape(-1))
    reward_loss = ce(reward_logits, target_rewards)
    done_loss = nn.functional.binary_cross_entropy_with_logits(
        done_logits, target_dones
    )
    return board_loss + reward_loss + done_loss


def train(model, x, boards_t, rewards_t, dones_t, epochs=20, batch_size=512):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(x)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            loss = compute_loss(
                model, x[idx], boards_t[idx], rewards_t[idx], dones_t[idx]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"epoch {epoch}: loss {total / n:.4f}")
