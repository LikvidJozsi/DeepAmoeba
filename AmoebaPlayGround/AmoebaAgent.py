import math
import random
from typing import List

import numpy as np

from AmoebaPlayGround.GameBoard import AmoebaBoard, Symbol
from AmoebaPlayGround.RewardCalculator import TrainingSample


class AmoebaAgent:
    def get_step(self, game_boards):
        pass

    def train(self, training_samples: List[TrainingSample]):
        pass

    def save(self, model_name):
        pass

    def get_name(self):
        return 'Default Name'


class ConsoleAgent(AmoebaAgent):
    def get_step(self, game_boards):
        answers = np.zeros((len(game_boards), 2), dtype='int')
        print('Give position in row column format (zero indexing):')
        for index, game_board in enumerate(game_boards):
            answer = input().split(' ')
            answers[index] = int(answer[0]), int(answer[1])
        return answers


# Random agent makes random (but relatively sensible plays) it is mainly for testing purposes, but may be incorporeted into early network training too.
# Play selection is done by determining cells that are at maximum 2 cells (configurable) away from an already placed symbol and choosing from them using an uniform distribution
class RandomAgent(AmoebaAgent):
    def __init__(self, move_max_distance=2):
        self.max_move_distance = move_max_distance

    def get_step(self, game_boards):
        steps = []
        for game_board in game_boards:
            steps.append(self.get_step_for_game(game_board))
        return steps

    def get_step_for_game(self, game_board: AmoebaBoard):
        eligible_cells = self.get_eligible_cells(game_board)
        if len(eligible_cells) == 0:
            if game_board.is_cell_empty(game_board.get_middle_of_map_index()):
                return game_board.get_middle_of_map_index()
            else:
                raise Exception('There are no free cells')
        chosen_cell = math.floor(random.uniform(0, len(eligible_cells)))
        return eligible_cells[chosen_cell]

    def get_eligible_cells(self, game_board: AmoebaBoard):
        eligible_cells = []
        for row_index, row in enumerate(game_board):
            for column_index, cell in enumerate(row):
                if cell == Symbol.EMPTY and self.has_close_symbol(game_board, row_index, column_index):
                    eligible_cells.append((row_index, column_index))
        return eligible_cells

    def has_close_symbol(self, game_board: AmoebaBoard, start_row, start_column):
        for row in range(start_row - self.max_move_distance, start_row + self.max_move_distance):
            for column in range(start_column - self.max_move_distance, start_column + self.max_move_distance):
                if game_board.is_within_bounds((row, column)) and not game_board.is_cell_empty((row, column)):
                    return True
        return False

    def get_name(self):
        return 'RandomAgent'
