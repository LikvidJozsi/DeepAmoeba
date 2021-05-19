import copy
from typing import List

from AmoebaPlayGround.GameBoard import AmoebaBoard, Symbol, Player

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
        self.map = AmoebaBoard(map_size)
        self.reset()

    def get_board_of_previous_player(self):
        return self.map.get_numeric_representation_for_player(self.previous_player)

    def init_map(self):
        self.map.reset()
        self.place_initial_symbol()

    def place_initial_symbol(self):
        self.map.set(self.map.get_middle_of_map_index(), Symbol.X)

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
            map.set(step, Symbol.EMPTY)
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
        y = move[0]
        x = move[1]
        player_symbol = game_board.get(move)
        player_won = (
                AmoebaGame.is_there_winning_line_in_direction(game_board, player_symbol,
                                                              y_start=y - win_sequence_length + 1,
                                                              x_start=x,
                                                              y_direction=1, x_direction=0) or  # vertical
                AmoebaGame.is_there_winning_line_in_direction(game_board, player_symbol,
                                                              y_start=y - win_sequence_length + 1,
                                                              x_start=x - win_sequence_length + 1,
                                                              y_direction=1, x_direction=1) or  # diagonal1
                AmoebaGame.is_there_winning_line_in_direction(game_board, player_symbol, y_start=y,
                                                              x_start=x - win_sequence_length + 1,
                                                              y_direction=0, x_direction=1) or  # horizontal
                AmoebaGame.is_there_winning_line_in_direction(game_board, player_symbol,
                                                              y_start=y + win_sequence_length - 1,
                                                              x_start=x - win_sequence_length + 1,
                                                              y_direction=-1, x_direction=1))  # diagonal2
        is_draw = AmoebaGame.is_map_full()
        return player_won, is_draw

    @staticmethod
    def is_map_full(game_board):
        return not Symbol.EMPTY in game_board

    @staticmethod
    def is_there_winning_line_in_direction(game_board, player_symbol, y_start, x_start, y_direction, x_direction):
        # ....x....
        # only 4 places in each direction count in determining if the new move created a winning condition of
        # a five figure long line
        search_length = 9
        line_length = 0
        for line_index in range(0, search_length):
            # depending on the direction of the line being searched direction may be 0 meaning the coordinate does
            # not change on any iterations,
            x_offset = line_index * x_direction
            y_offset = line_index * y_direction
            y = y_start + y_offset
            x = x_start + x_offset
            if game_board.is_within_bounds((y, x)) and game_board.get((y, x)) == player_symbol:
                line_length += 1
            else:
                line_length = 0
            if line_length == win_sequence_length:
                return True
        return False
