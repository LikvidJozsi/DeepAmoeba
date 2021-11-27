import json
import math
import random

import numpy as np

from AmoebaPlayGround import Amoeba
from AmoebaPlayGround.Agents.AmoebaAgent import AmoebaAgent
from AmoebaPlayGround.Agents.MCTS.MCTSAgent import MCTSAgent
from AmoebaPlayGround.Amoeba import AmoebaGame
from AmoebaPlayGround.GameBoard import Player
from AmoebaPlayGround.GameExecution.MoveSelector import MaximalMoveSelector


class Puzzle:

    def __init__(self, json_representation, variation_count_target):
        self.board_state = np.array(json_representation["board_state"])
        self.solutions = json_representation["correct_moves"]
        self.extra_prohibited_filler_places = json_representation["extra_prohibited_filler_places"]
        self.symmetry_period = json_representation["symmetry_period"]
        self.value_expected = json_representation["value_expected"]
        self.board_state_variations = []
        self.solution_variations = []
        self.generate_variations(variation_count_target)

    def generate_variations(self, variation_count_target):
        map_size = np.array(Amoeba.map_size)
        rotated_state = self.board_state
        rotated_solutions = self.solutions
        rotated_prohibited_spaces = self.extra_prohibited_filler_places
        for rotation in range(0, self.symmetry_period, 90):
            rotated_state = np.rot90(rotated_state)
            rotated_solutions = self.rotate_indexes_90_deg(rotated_solutions, rotated_state.shape)
            rotated_prohibited_spaces = self.rotate_indexes_90_deg(rotated_prohibited_spaces, rotated_state.shape)
            translation_room = map_size - rotated_state.shape
            for x_translation in range(translation_room[0]):
                for y_translation in range(translation_room[1]):
                    translation = [x_translation, y_translation]
                    translated_prohibited_spaces = self.translate_indexes(rotated_prohibited_spaces, translation)
                    full_size_board = self.get_translated_board_state(rotated_state, translation, map_size)
                    translated_solutions = self.translate_indexes(rotated_solutions, translation)
                    self.equalize_pieces(full_size_board, translated_solutions, translated_prohibited_spaces)
                    self.board_state_variations.append(full_size_board)
                    self.solution_variations.append(translated_solutions)
        self.reduce_variation_count(variation_count_target)

    def reduce_variation_count(self, target):
        variation_count = len(self.board_state_variations)
        if variation_count < target:
            return
        selection = np.random.choice(variation_count, size=target, replace=False)
        filtered_board_states = []
        filtered_solutions = []
        for selected in selection:
            filtered_board_states.append(self.board_state_variations[selected])
            filtered_solutions.append(self.solution_variations[selected])
        self.board_state_variations = filtered_board_states
        self.solution_variations = filtered_solutions

    def get_translated_board_state(self, board_state, translation, map_size):
        translated_board_state = np.zeros(map_size, dtype=np.int8)
        translated_board_state[translation[0]:(translation[0] + board_state.shape[0]),
        translation[1]:(translation[1] + board_state.shape[1])] = board_state
        return AmoebaGame(board_state=translated_board_state)

    def equalize_pieces(self, board_state, solutions, prohibited_spaces):
        own_figure_count = np.count_nonzero(board_state.map.cells == 1)
        other_figure_count = np.count_nonzero(board_state.map.cells == -1)
        max_figure_count = max(own_figure_count, other_figure_count)
        board_size = board_state.map.get_shape()
        for i in range(max_figure_count - own_figure_count):
            while True:
                x = random.randrange(0, board_size[0])
                y = random.randrange(0, board_size[1])
                if board_state.map.get((x, y)) == 0 and not is_move_in_move_list(solutions,
                                                                                 [x, y]) and not is_move_in_move_list(
                    prohibited_spaces, [x, y]):
                    board_state.map.set((x, y), 1)
                    break

        for i in range(max_figure_count - other_figure_count):
            while True:
                x = random.randrange(0, board_size[0])
                y = random.randrange(0, board_size[1])
                if board_state.map.get((x, y)) == 0 and not is_move_in_move_list(solutions,
                                                                                 [x, y]) and not is_move_in_move_list(
                    prohibited_spaces, [x, y]):
                    board_state.map.set((x, y), -1)
                    break

    def translate_indexes(self, indexes, translation):
        return [[index[0] + translation[0], index[1] + translation[1]] for index in indexes]

    def rotate_indexes_90_deg(self, indexes, board_size):
        return [[board_size[0] - 1 - x[1], x[0]] for x in indexes]


class PuzzleEvaluator:
    def __init__(self, variation_count_target):
        self.move_selector = MaximalMoveSelector()
        with open("Puzzles/level_1.json", "r") as file:
            self.level_1 = self.load_puzzles(json.load(file), variation_count_target)

        with open("Puzzles/level_2.json", "r") as file:
            self.level_2 = self.load_puzzles(json.load(file), variation_count_target)

        with open("Puzzles/level_3.json", "r") as file:
            self.level_3 = self.load_puzzles(json.load(file), variation_count_target)

        with open("Puzzles/level_4.json", "r") as file:
            self.level_4 = self.load_puzzles(json.load(file), variation_count_target)

        with open("Puzzles/level_5.json", "r") as file:
            self.level_5 = self.load_puzzles(json.load(file), variation_count_target)

    def load_puzzles(self, json_representation, variation_count_target):
        return [Puzzle(json_repr, variation_count_target) for json_repr in json_representation]

    def evaluate_agent(self, agent: AmoebaAgent, logger=None):
        self.evaluate_on_level(agent, logger, "level_1", self.level_1)
        self.evaluate_on_level(agent, logger, "level_2", self.level_2)
        self.evaluate_on_level(agent, logger, "level_3", self.level_3)
        self.evaluate_on_level(agent, logger, "level_4", self.level_4)
        self.evaluate_on_level(agent, logger, "level_5", self.level_5)

    def evaluate_on_level(self, agent, logger, level_name, level_puzzles):
        policy_score, policy_entropy, search_score, search_entropy, \
        value_error = self.evaluate_on_puzzles(agent, level_puzzles)
        logger.log(level_name + "_puzzle_policy_score", policy_score)
        logger.log(level_name + "_puzzle_policy_entropy", policy_entropy)
        logger.log(level_name + "_puzzle_value_error", value_error)
        logger.log(level_name + "_puzzle_search_score", search_score)
        logger.log(level_name + "_puzzle_search_entropy", search_entropy)
        print(("{0} puzzle result: \n\tpolicy: score {1:.03f}, entropy {2:.03f}\n\tsearch: score {3:.03f}, " +
               "entropy {4:.03f}, value error: {5:.03f}").format(
            level_name, policy_score, policy_entropy, search_score, search_entropy, value_error
        ))

    def evaluate_on_puzzles(self, agent: AmoebaAgent, puzzles):
        aggreage_search_correctness, aggregate_search_entropy = 0, 0
        aggregate_policy_correctness, aggregate_policy_entropy = 0, 0
        aggregate_value_error, value_error_counts = 0, 0
        for index, puzzle in enumerate(puzzles):
            policy_correctness, policy_entropy, value_error = self.evaluate_policy(agent, puzzle)
            if value_error is not None:
                aggregate_value_error += value_error
                value_error_counts += 1
            aggregate_policy_correctness += policy_correctness
            aggregate_policy_entropy += policy_entropy
            search_correctness, search_entropy = self.evaluate_puzzle(agent, puzzle)
            aggreage_search_correctness += search_correctness
            aggregate_search_entropy += search_entropy

        average_search_correctness = aggreage_search_correctness / len(puzzles)
        average_search_entropy = aggregate_search_entropy / len(puzzles)

        average_policy_correctness = aggregate_policy_correctness / len(puzzles)
        average_policy_entropy = aggregate_policy_entropy / len(puzzles)

        if value_error_counts > 0:
            average_value_error = aggregate_value_error / value_error_counts
        else:
            average_value_error = 0

        return average_policy_correctness, average_policy_entropy, average_search_correctness, \
               average_search_entropy, average_value_error

    def evaluate_policy(self, agent: MCTSAgent, puzzle):
        raw_boards = [board.map.cells for board in puzzle.board_state_variations]
        formatted_input = agent.format_input(raw_boards)
        output_1d, values = agent.model.predict(formatted_input, batch_size=len(raw_boards))
        board_size = raw_boards[0].shape
        output_2d = output_1d.reshape(-1, board_size[0], board_size[1])
        correct_moves = self.get_correct_move_count(output_2d, puzzle)

        variation_count = len(output_2d)
        correctness = correct_moves / variation_count
        entropy = self.get_entropy(correctness)

        if puzzle.value_expected is not None:
            puzzle_error = np.sum(np.power(values - puzzle.value_expected, 2)) / variation_count
        else:
            puzzle_error = None
        return correctness, entropy, puzzle_error

    def evaluate_puzzle(self, agent: AmoebaAgent, puzzle):
        probabilities, _ = agent.get_step(puzzle.board_state_variations, Player.X, False)

        correct_moves = self.get_correct_move_count(probabilities, puzzle)
        variation_count = len(probabilities)
        correctness = correct_moves / variation_count
        entropy = self.get_entropy(correctness)
        return correctness, entropy

    def get_correct_move_count(self, probabilities, puzzle):
        correct_moves = 0
        for probability_map, solution_variations in zip(probabilities, puzzle.solution_variations):
            chosen_move = self.move_selector.select_move(probability_map)
            if is_move_in_move_list(solution_variations, chosen_move):
                correct_moves += 1
        return correct_moves

    def get_entropy(self, correctness):
        if correctness != 0 and correctness != 1:
            entropy = -correctness * math.log(correctness, 2) - (1 - correctness) * math.log(1 - correctness, 2)
        else:
            entropy = 0
        return entropy


def is_move_in_move_list(list, move):
    for element in list:
        if move[0] == element[0] and move[1] == element[1]:
            return True
    return False
