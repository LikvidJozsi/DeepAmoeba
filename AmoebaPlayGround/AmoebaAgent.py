from typing import List

import numpy as np

from AmoebaPlayGround.GameBoard import AmoebaBoard, Symbol
from AmoebaPlayGround.TrainingSampleGenerator import TrainingSample


class AmoebaAgent:
    def get_step(self, game_boards: List[AmoebaBoard], player) -> List[np.ndarray[np.float32]]:
        pass

    def train(self, training_samples: List[TrainingSample]):
        pass

    def save(self, model_name):
        pass

    def get_name(self):
        return 'Default Name'


class ConsoleAgent(AmoebaAgent):
    def get_step(self, game_boards: List[AmoebaBoard], player):
        answers = np.zeros((len(game_boards), 2), dtype=np.uint8)
        print('You are ' + str(player))
        print('Give position in row column format (zero indexing):')
        for index, game_board in enumerate(game_boards):
            answer = input().split(' ')
            probabilities = np.zeros(game_board.get_shape(), dtype=np.float32)
            probabilities[int(answer[0]), int(answer[1])] = 1
            answers[index] = probabilities
        return answers


# Random agent makes random (but relatively sensible plays) it is mainly for testing purposes, but may be incorporeted into early network training too.
# Play selection is done by determining cells that are at maximum 2 cells (configurable) away from an already placed symbol and choosing from them using an uniform distribution
class RandomAgent(AmoebaAgent):
    def __init__(self, move_max_distance=2):
        self.max_move_distance = move_max_distance

    def get_step(self, game_boards: List[AmoebaBoard], player) -> List[np.ndarray[np.float32]]:
        steps = []
        for game_board in game_boards:
            steps.append(self.get_eligible_cells(game_board))
        return steps

    def get_eligible_cells(self, game_board: AmoebaBoard):
        eligible_cells = np.zeros(game_board.get_shape(), dtype=np.float32)
        for row_index, row in enumerate(game_board):
            for column_index, cell in enumerate(row):
                if cell == Symbol.EMPTY and self.has_close_symbol(game_board, row_index, column_index):
                    eligible_cells[row_index, column_index] = 1
        return eligible_cells

    def has_close_symbol(self, game_board: AmoebaBoard, start_row, start_column):
        for row in range(start_row - self.max_move_distance, start_row + self.max_move_distance):
            for column in range(start_column - self.max_move_distance, start_column + self.max_move_distance):
                if game_board.is_within_bounds((row, column)) and not game_board.is_cell_empty((row, column)):
                    return True
        return False

    def get_name(self):
        return 'RandomAgent'
