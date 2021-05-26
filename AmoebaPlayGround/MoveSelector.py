import numpy as np


class MoveSelector:
    def select_move(self, probabilities: np.ndarray) -> tuple:
        pass


class MaximalMoveSelector(MoveSelector):
    def select_move(self, probabilities: np.ndarray) -> tuple:
        max_probability = np.max(probabilities)
        bestActions = np.array(np.argwhere(max_probability == probabilities))
        chosenActionIndex = np.random.randint(len(bestActions))
        return bestActions[chosenActionIndex]


class DistributionMoveSelector(MoveSelector):
    def select_move(self, probabilities: np.ndarray) -> tuple:
        indexes = np.arange(np.prod(probabilities.shape))
        choice = np.random.choice(indexes, p=probabilities.flatten())
        choice_multi_d_index = np.unravel_index([choice], probabilities.shape)
        return choice_multi_d_index[0][0], choice_multi_d_index[1][0]
