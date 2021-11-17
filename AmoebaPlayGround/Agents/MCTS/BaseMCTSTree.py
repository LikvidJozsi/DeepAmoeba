import numpy as np

from AmoebaPlayGround.Amoeba import AmoebaGame
from AmoebaPlayGround.GameBoard import AmoebaBoard, EMPTY_SYMBOL


class MCTSNode:
    def __init__(self, board_state: AmoebaBoard, has_game_ended=False, **kwargs):
        self.board_state: AmoebaBoard = board_state
        map_size = board_state.get_shape()
        self.sum_expected_move_rewards: np.ndarray[np.float32] = np.zeros(map_size, dtype=np.float32)
        self.forward_visited_counts: np.ndarray[np.uint16] = np.zeros(map_size, dtype=np.uint16)
        self.backward_visited_counts: np.ndarray[np.uint16] = np.zeros(map_size, dtype=np.uint16)
        self.invalid_moves = board_state.cells != EMPTY_SYMBOL
        self.visited_count = 0
        self.neural_network_policy = None
        self.game_has_ended = has_game_ended
        self.reward = 0
        self.pending_policy_calculation = False

    def set_game_ended(self, move):
        player_won, is_draw = AmoebaGame.check_game_ended(self.board_state, move)
        # reward from the perspective of the next player
        # if previous player won, its bad for next player
        if player_won:
            self.reward = -1
            self.game_has_ended = True
        if is_draw:
            # reward for draw is 0, but it is subject to change
            self.game_has_ended = True
            self.reward = 0

    def policy_calculation_started(self):
        self.pending_policy_calculation = True

    def policy_calculation_ended(self):
        self.pending_policy_calculation = False

    def set_policy(self, policy):
        self.neural_network_policy = policy

    def is_unvisited(self):
        return self.neural_network_policy is None

    def get_board_state_after_move(self, move, player):
        new_board_state = self.board_state.copy()
        new_board_state.set(move, player.get_symbol())
        return new_board_state

    def move_forward_selected(self, move):
        self.forward_visited_counts[move] += 1
        self.visited_count += 1

    def has_game_ended(self):
        return self.game_has_ended

    def add_virtual_loss(self, move, virtual_loss):
        self.sum_expected_move_rewards[move] -= virtual_loss
        self.backward_visited_counts[move] += virtual_loss

    def update_expected_value_for_move(self, move, simulation_value, virtual_loss_to_remove):
        self.sum_expected_move_rewards[move] += simulation_value + virtual_loss_to_remove
        self.backward_visited_counts[move] += 1 - virtual_loss_to_remove

    def get_policy(self):
        return self.neural_network_policy


class MCTSRootNode(MCTSNode):
    def __init__(self, board_state: AmoebaBoard, has_game_ended=False, eps=0.25, **kwargs):
        super().__init__(board_state=board_state, has_game_ended=has_game_ended, **kwargs)
        self.eps = eps

    def set_policy(self, policy):
        board_shape = self.board_state.get_shape()
        self.neural_network_policy = policy * (1 - self.eps) + self.eps * np.random.dirichlet(
            [0.06] * np.prod(board_shape)).reshape(board_shape)

    def get_policy(self):
        return self.neural_network_policy


class BaseMCTSTree:

    def get_the_node_of_move(self, search_node, move, player):
        pass

    def set_turn(self, turn):
        pass

    def get_root_node(self, game, eps):
        pass

    def get_node_count(self):
        pass
