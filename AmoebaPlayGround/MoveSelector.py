from typing import List

import numpy as np

class MoveSelector:
    def select_move(self,probabilities: np.ndarray[np.float32]) -> tuple:
        pass


class MaximalMoveSelector(MoveSelector):
    def select_move(self,probabilities: np.ndarray[np.float32]) -> tuple:
        indexes = np.arange(np.prod(probabilities.shape))
        choice = np.random.choice(indexes, p=probabilities.flatten())
        choice_multi_d_index = np.unravel_index([choice], probabilities.shape)
        return choice_multi_d_index[0][0],choice_multi_d_index[1][0]