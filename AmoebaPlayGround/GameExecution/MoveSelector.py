import numpy as np


class MoveSelector:
    def select_move(self, probabilities: np.ndarray) -> tuple:
        pass


class MaximalMoveSelector(MoveSelector):
    def select_move(self, probabilities: np.ndarray) -> tuple:
        max_probability = np.max(probabilities)
        bestActions = np.array(np.argwhere(max_probability == probabilities))
        chosenActionIndex = np.random.randint(len(bestActions))
        return tuple(bestActions[chosenActionIndex])


class DistributionMoveSelector(MoveSelector):
    def __init__(self, temperature=1 / 8):
        self.temperature = temperature

    def select_move(self, probabilities: np.ndarray) -> tuple:
        probabilities = self.apply_temperature(probabilities, self.temperature)

        indexes = np.arange(np.prod(probabilities.shape))
        choice = np.random.choice(indexes, p=probabilities.flatten())
        choice_multi_d_index = np.unravel_index([choice], probabilities.shape)
        return choice_multi_d_index[0][0], choice_multi_d_index[1][0]

    def apply_temperature(self, probabilities, temperature):
        temperature_agumented_probabilities = np.power(probabilities, 1 / temperature)
        temperature_agumented_probabilities = temperature_agumented_probabilities / np.sum(
            temperature_agumented_probabilities)
        return temperature_agumented_probabilities


class MoveSelectionStrategy:

    def __init__(self, early_game_move_selector=DistributionMoveSelector(1),
                 late_game_move_selector=DistributionMoveSelector(1 / 16), late_game_start_turn=5):
        self.early_game_move_selector = early_game_move_selector
        self.late_game_move_selector = late_game_move_selector
        self.late_game_start_turn = late_game_start_turn

    def get_move_selector(self, turn):
        if turn >= self.late_game_start_turn:
            return self.late_game_move_selector
        else:
            return self.early_game_move_selector


class EvaluationMoveSelectionStrategy:
    def __init__(self, early_game_move_selector=DistributionMoveSelector(1/4),
                 late_game_move_selector=DistributionMoveSelector(1 / 16), late_game_start_turn=15):
        self.early_game_move_selector = early_game_move_selector
        self.late_game_move_selector = late_game_move_selector
        self.late_game_start_turn = late_game_start_turn

    def get_move_selector(self, turn):
        if turn >= self.late_game_start_turn:
            return self.late_game_move_selector
        else:
            return self.early_game_move_selector


MOVE_SELECTION_STRATEGIES = {
    "MoveSelectionStrategy": MoveSelectionStrategy(),
    "EvaluationMoveSelectionStrategy": EvaluationMoveSelectionStrategy()
}
