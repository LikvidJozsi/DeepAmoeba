import random
from typing import List

import numpy as np

from AmoebaPlayGround.Amoeba import AmoebaGame
from AmoebaPlayGround.GameBoard import AmoebaBoard, EMPTY_SYMBOL
from AmoebaPlayGround.Training.Logger import Statistics


class AmoebaAgent:
    def get_step(self, games: List[AmoebaGame], player, evaluation=False):
        pass

    def train(self, training_samples: List):
        pass

    def save(self, model_name):
        pass

    def get_name(self):
        return 'Default Name'

    def reset(self):
        pass

    def get_random_start(self, board):
        map_size = board.get_shape()
        middle = (int(map_size[0] / 2), int(map_size[1] / 2))
        move = (random.randint(-1, 1) + middle[0], random.randint(-1, 1) + middle[1])
        probability_map = np.zeros(map_size, dtype=np.float32)
        probability_map[move] = 1
        return probability_map


class PlaceholderAgent(AmoebaAgent):
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


class ConsoleAgent(AmoebaAgent):
    def get_step(self, games: List[AmoebaGame], player, evaluation=False):
        game_boards = [game.map for game in games]
        answers = np.zeros((len(game_boards), 2), dtype=np.int8)
        print('You are ' + str(player))
        print('Give position in row column format (zero indexing):')
        for index, game_board in enumerate(game_boards):
            answer = input().split(' ')
            probabilities = np.zeros(game_board.get_shape(), dtype=np.float32)
            probabilities[int(answer[0]), int(answer[1])] = 1
            answers[index] = probabilities
        return answers, Statistics()


# Random agent makes random (but relatively sensible plays) it is mainly for testing purposes, but may be incorporeted into early network training too.
# Play selection is done by determining cells that are at maximum 2 cells (configurable) away from an already placed symbol and choosing from them using an uniform distribution
class RandomAgent(AmoebaAgent):
    def __init__(self, move_max_distance=2):
        self.max_move_distance = move_max_distance

    def get_step(self, games: List[AmoebaGame], player, evaluation=False):
        steps = []
        for game in games:
            if game.num_steps > 0:
                steps.append(self.get_move_probabilities(game.map))
            else:
                steps.append(self.get_random_start(game.map))
        return steps, Statistics()

    def get_move_probabilities(self, game_board: AmoebaBoard):
        eligible_cells = np.zeros(game_board.get_shape(), dtype=np.float32)
        for row_index, row in enumerate(game_board):
            for column_index, cell in enumerate(row):
                if cell == EMPTY_SYMBOL and self.has_close_symbol(game_board, row_index, column_index):
                    eligible_cells[row_index, column_index] = 1
        probabilities = eligible_cells / np.sum(eligible_cells)
        return probabilities

    def has_close_symbol(self, game_board: AmoebaBoard, start_row, start_column):
        for row in range(start_row - self.max_move_distance, start_row + self.max_move_distance):
            for column in range(start_column - self.max_move_distance, start_column + self.max_move_distance):
                if game_board.is_within_bounds((row, column)) and not game_board.is_cell_empty((row, column)):
                    return True
        return False

    def get_name(self):
        return 'RandomAgent'
