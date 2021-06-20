import copy
from typing import List

from AmoebaPlayGround.GameBoard import AmoebaBoard, Player, X_SYMBOL, EMPTY_SYMBOL

win_sequence_length = 5
map_size = (8, 8)


class Move:
    def __init__(self, board_state, step, player: Player):
        self.board_state = board_state
        self.step = step
        self.player = player


class AmoebaGame:
    def __init__(self, view=None):
        self.view = view
        if len(map_size) != 2:
            raise Exception('Map must be two dimensional but found shape %s' % (map_size))
        if win_sequence_length >= map_size[0]:
            raise Exception('Map size is smaller than the length of a winning sequence.')
        self.map = AmoebaBoard(size=map_size)
        self.reset()

    def get_board_of_previous_player(self):
        return self.map.get_numeric_representation_for_player(self.previous_player)

    def init_map(self):
        self.map.reset()
        self.place_initial_symbol()

    def get_next_player(self):
        return self.previous_player.get_other_player()

    def place_initial_symbol(self):
        self.map.set(self.map.get_middle_of_map_index(), X_SYMBOL)

    def reset(self):
        self.init_map()
        self.previous_player = Player.X
        self.history = []
        self.winner = None
        self.num_steps = 1
        if self.view is not None:
            self.view.display_game_state(self.map)

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

    def has_game_ended(self):
        last_action = self.history[-1]
        player_won, is_draw = AmoebaGame.check_game_ended(self.map, last_action)
        if player_won:
            self.winner = self.previous_player
        if is_draw:
            self.winner = Player.NOBODY
        if (player_won or is_draw) and self.view is not None:
            self.view.game_ended(self.winner)
        return is_draw or player_won

    @staticmethod
    def check_game_ended(game_board: AmoebaBoard, move):
        player_symbol = game_board.get(move)
        player_won = (
                AmoebaGame.is_there_winning_line_in_direction(game_board, player_symbol, move, [1, 0]) or  # vertical
                AmoebaGame.is_there_winning_line_in_direction(game_board, player_symbol, move, [1, 1]) or  # diagonal1
                AmoebaGame.is_there_winning_line_in_direction(game_board, player_symbol, move, [0, 1]) or  # horizontal
                AmoebaGame.is_there_winning_line_in_reverse_diagonal(game_board, player_symbol, move))  # diagonal2
        is_draw = False
        if not player_won:
            is_draw = AmoebaGame.is_map_full(game_board)
        return player_won, is_draw

    @staticmethod
    def is_map_full(game_board):
        return not EMPTY_SYMBOL in game_board.cells

    @staticmethod
    def is_there_winning_line_in_direction(game_board, player_symbol, move, dir_vector):
        # ....x....
        # only 4 places in each direction count in determining if the new move created a winning condition of
        # a five figure long line

        max_distance = win_sequence_length - 1
        max_x_neg_offset = AmoebaGame.get_maximum_negative_offset(move[0], dir_vector[0], max_distance)
        max_y_neg_offset = AmoebaGame.get_maximum_negative_offset(move[1], dir_vector[1], max_distance)
        max_neg_offset = min(max_x_neg_offset, max_y_neg_offset)

        board_size = game_board.get_shape()
        max_x_pos_offset = AmoebaGame.get_maximum_positive_offset(move[0], dir_vector[0], max_distance, board_size[0])
        max_y_pos_offset = AmoebaGame.get_maximum_positive_offset(move[1], dir_vector[1], max_distance, board_size[1])
        max_pos_offset = min(max_x_pos_offset, max_y_pos_offset)

        line_length = 0
        for offset in range(-max_neg_offset, max_pos_offset + 1):
            # depending on the direction of the line being searched direction may be 0 meaning the coordinate does
            # not change on any iterations
            x = move[0] + offset * dir_vector[0]
            y = move[1] + offset * dir_vector[1]
            if game_board.get((x, y)) == player_symbol:
                line_length += 1
            else:
                line_length = 0
            if max_pos_offset - offset + line_length < max_distance + 1:
                return False
            if line_length == win_sequence_length:
                return True
        return False

    @staticmethod
    def is_there_winning_line_in_reverse_diagonal(game_board, player_symbol, move):
        # in the reverse diagonal
        # ........x
        # .......x.
        # ......x..
        # .....x...
        # ....x....
        # one of the coordinates increases while to other decreases, so the boundary conditions get switched up

        board_size = game_board.get_shape()
        max_distance = win_sequence_length - 1
        max_x_neg_offset = AmoebaGame.get_maximum_negative_offset(move[0], 1, max_distance)
        max_y_pos_offset = AmoebaGame.get_maximum_positive_offset(move[1], 1, max_distance, board_size[1])
        max_neg_offset = min(max_x_neg_offset, max_y_pos_offset)

        max_x_pos_offset = AmoebaGame.get_maximum_positive_offset(move[0], 1, max_distance, board_size[0])
        max_y_neg_offset = AmoebaGame.get_maximum_negative_offset(move[1], 1, max_distance)
        max_pos_offset = min(max_x_pos_offset, max_y_neg_offset)

        line_length = 0
        for offset in range(-max_neg_offset, max_pos_offset + 1):
            # depending on the direction of the line being searched direction may be 0 meaning the coordinate does
            # not change on any iterations,
            x = move[0] + offset
            y = move[1] - offset
            if game_board.get((x, y)) == player_symbol:
                line_length += 1
            else:
                line_length = 0
            if max_pos_offset - offset + line_length < max_distance + 1:
                return False
            if line_length == win_sequence_length:
                return True
        return False

    @staticmethod
    def get_maximum_negative_offset(move_coordinate, line_direction, max_search_distance):
        return max_search_distance - max(max_search_distance * line_direction - move_coordinate, 0)

    @staticmethod
    def get_maximum_positive_offset(move_coordinate, line_direction, max_search_distance, map_length):
        return max_search_distance - max(max_search_distance * line_direction + move_coordinate - map_length + 1, 0)
