import numpy as np


class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1
        self.done = False
        return self.observation()

    def observation(self):
        return np.append(self.board, self.current_player).astype(np.int8)

    def legal_actions(self):
        return np.where(self.board == 0)[0]

    def step(self, action):
        assert self.current_player in (1, -1)
        assert not self.done, "Game finished, call reset()"
        assert self.board[action] == 0, f"cell {action} is occupied"

        self.board[action] = self.current_player

        reward = 0

        if self._check_win(self.current_player):
            reward = self.current_player
            self.done = True
        elif len(self.legal_actions()) == 0:
            self.done = True

        self.current_player = -self.current_player
        return self.observation(), reward, self.done

    WIN_LINES = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ]

    def _check_win(self, current_player):
        for lines in self.WIN_LINES:
            finish = True
            for idx in lines:
                if self.board[idx] != current_player:
                    finish = False
                    break
            if finish:
                return True
        return False
