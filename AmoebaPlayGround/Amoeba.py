import copy
import random
from typing import List

import numpy as np

from AmoebaPlayGround.GameBoard import AmoebaBoard, Player, X_SYMBOL, EMPTY_SYMBOL
from AmoebaPlayGround.GameEndChecker import check_victory_condition
from AmoebaPlayGround.GameExecution.MoveSelector import MaximalMoveSelector

win_sequence_length = 5


class Move:
    def __init__(self, board_state, step, player: Player):
        self.board_state = board_state
        self.step = step
        self.player = player


class AmoebaGame:
    def __init__(self, map_size, view=None, board_state=None, play_first_move=True):
        self.view = view
        if len(map_size) != 2:
            raise Exception('Map must be two dimensional but found shape %s' % (str(map_size)))
        if win_sequence_length >= map_size[0]:
            raise Exception('Map size is smaller than the length of a winning sequence.')
        self.map = AmoebaBoard(size=map_size, cells=board_state)
        self.num_steps = 0
        self.history = []
        self.additional_data = {}
        if board_state is None:
            self.init_map()
            if play_first_move:
                self.place_first_piece()
                self.previous_player = Player.X
                self.num_steps = 1
            else:
                self.previous_player = Player.O

        self.winner = None

        if self.view is not None:
            self.view.display_game_state(self.map)

    def clear_additional_data(self):
        self.additional_data = {}

    def additional_data_present(self, key):
        return key in self.additional_data

    def set_additional_data(self, key, data):
        self.additional_data[key] = data

    def get_additional_data(self, key):
        return self.additional_data[key]

    def place_first_piece(self, max_distance_from_center=2):
        board_size = self.map.get_shape()
        board_half_size = board_size[0] // 2
        max_distance_from_center = min(max_distance_from_center, board_half_size - 1)

        column_offset = random.randint(-max_distance_from_center, max_distance_from_center)
        row_offset = random.randint(-max_distance_from_center, max_distance_from_center)

        column = board_half_size + column_offset
        row = board_half_size + row_offset
        self.map.set((column, row), X_SYMBOL)
        self.history.append((column, row))

    def play_game(self, x_agent, o_agent, hightlight_agent=None):
        agents = [x_agent, o_agent]
        current_agent_index = 0
        move_selector = MaximalMoveSelector()
        while (not self.has_game_ended()):
            current_agent = agents[current_agent_index]

            if hightlight_agent is not None:
                output_1d, value = hightlight_agent.model.predict([self.map.cells], [self.get_next_player()])
                board_size = self.map.get_shape()
                output_2d = output_1d.reshape(-1, board_size[0], board_size[1])
                color_intensities = np.array(output_2d[0] / np.max(output_2d[0]) * 255, dtype=int)
                self.view.display_background_color_intensities(color_intensities)
                self.view.set_additional_info(f"value: {value[0][0]}")
            action_probabilities, step_statistics = current_agent.get_step([self], self.get_next_player())
            action = move_selector.select_move(action_probabilities[0])
            print(action)
            self.step(action)
            current_agent_index = (current_agent_index + 1) % 2

    def get_board_of_next_player(self):
        return self.map.get_numeric_representation_for_player(self.get_next_player())

    def init_map(self):
        self.map.reset()

    def get_next_player(self):
        return self.previous_player.get_other_player()

    def place_initial_symbol(self):
        self.map.set(self.map.get_middle_of_map_index(), X_SYMBOL)

    def step(self, action):
        current_player = self.previous_player.get_other_player()
        player_symbol = current_player.get_symbol()
        action = tuple(action)
        if not self.map.is_cell_empty(action):
            raise Exception('Trying to place symbol in position already occupied')
        self.map.set(action, player_symbol)
        self.history.append(action)
        self.num_steps += 1
        self.previous_player = current_player
        if self.view is not None:
            self.view.display_game_state(self.map)

    def get_last_moves(self, number_of_steps: int) -> List[Move]:
        moves = []
        map = copy.deepcopy(self.map)
        if len(self.history) < number_of_steps:
            number_of_steps = len(self.history)

        player = self.previous_player
        for index in range(number_of_steps):
            step = self.history[len(self.history) - index - 1]
            map.set(step, EMPTY_SYMBOL)
            move = Move(copy.deepcopy(map), step, player)
            moves.append(move)
            player = player.get_other_player()
        return moves

    def get_moves_since_turn(self, starting_turn_index):
        return self.history[starting_turn_index:self.num_steps]

    def has_game_ended(self):
        if len(self.history) == 0:
            return False
        last_action = self.history[-1]
        player_won = check_victory_condition(self.map.cells, np.array(last_action, dtype=int))
        is_draw = self.map.occupied_cells == np.prod(self.map.cells.shape)
        if player_won:
            self.winner = self.previous_player
        if is_draw:
            self.winner = Player.NOBODY
        if (player_won or is_draw) and self.view is not None:
            self.view.game_ended(self.winner)
        return is_draw or player_won

    @staticmethod
    def check_game_ended(game_board: AmoebaBoard, move):
        return check_victory_condition(game_board.cells,
                                       np.array(move, dtype=int)), game_board.occupied_cells == np.prod(
            game_board.cells.shape)
