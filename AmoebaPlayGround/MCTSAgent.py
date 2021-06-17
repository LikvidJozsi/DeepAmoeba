from typing import List, Dict

import numpy as np

from AmoebaPlayGround.Amoeba import AmoebaGame
from AmoebaPlayGround.GameBoard import AmoebaBoard, EMPTY_SYMBOL
from AmoebaPlayGround.Logger import Statistics
from AmoebaPlayGround.NetworkModels import PolicyValueNetwork
from AmoebaPlayGround.NeuralAgent import NetworkModel, NeuralAgent
from AmoebaPlayGround.TrainingSampleGenerator import TrainingSampleCollection


class MCTSNode:
    def __init__(self, board_state: AmoebaBoard, has_game_ended=False):
        self.board_state: AmoebaBoard = board_state
        self.expected_move_rewards: np.ndarray[np.float32] = np.zeros(board_state.get_shape(), dtype=np.float32)
        self.forward_visited_counts: np.ndarray[np.uint16] = np.zeros(board_state.get_shape(), dtype=np.uint16)
        self.backward_visited_counts: np.ndarray[np.uint16] = np.zeros(board_state.get_shape(), dtype=np.uint16)
        self.invalid_moves = board_state.cells != EMPTY_SYMBOL
        self.invalid_move_count = np.sum(self.invalid_moves)
        self.visited_count = 0
        self.neural_network_policy = None
        self.game_has_ended = has_game_ended
        self.reward = None
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

    def update_expected_value_for_move(self, move, simulation_value):
        visited_count = self.backward_visited_counts[move]
        expected_reward = self.expected_move_rewards[move]
        self.expected_move_rewards[move] = (visited_count * expected_reward + simulation_value) / (
                visited_count + 1)
        self.backward_visited_counts[move] += 1


class MCTSAgent(NeuralAgent):

    def __init__(self, model_name=None, load_latest_model=False,
                 model_type: NetworkModel = PolicyValueNetwork(), search_count=100, exploration_rate=1.4,
                 training_epochs=10):
        super().__init__(model_type, model_name, load_latest_model)
        self.mcts_nodes: Dict[AmoebaBoard, MCTSNode] = {}
        self.search_count = search_count
        self.exploration_rate = exploration_rate
        self.training_epochs = training_epochs

    def reset(self):
        self.mcts_nodes = dict()

    def get_step(self, game_boards: List[AmoebaBoard], player):
        search_nodes = self.get_search_nodes_for_board_states(game_boards)
        for i in range(self.search_count):
            for node in search_nodes:
                self.run_search(node, player, set())
        return self.get_move_probabilities_from_nodes(search_nodes, player), Statistics()

    def get_search_nodes_for_board_states(self, game_boards):
        nodes = []
        for game_board in game_boards:
            board_copy = game_board.copy()
            search_node = self.get_search_node_of_state(board_copy)
            nodes.append(search_node)
        return nodes

    def get_move_probabilities_from_nodes(self, nodes, player):
        action_probabilities = []
        for node in nodes:
            action_visited_counts = node.forward_visited_counts
            probabilities = action_visited_counts / np.sum(action_visited_counts)
            action_probabilities.append(probabilities)
        return action_probabilities

    def get_search_node_of_state(self, board_state, move=None):
        search_node = self.mcts_nodes.get(board_state)
        if search_node is not None:
            return search_node
        else:
            new_node = MCTSNode(board_state)
            if move is not None:
                new_node.set_game_ended(move)
            self.mcts_nodes[board_state] = new_node
            return new_node

    def get_probability_distribution(self, search_node, player):
        game_board = search_node.board_state.cells
        formatted_input = self.format_input([game_board], [player])
        output_2d, value = self.model.predict(formatted_input, batch_size=1)
        output_2d = output_2d.reshape(game_board.shape)
        output_2d = output_2d * np.logical_not(search_node.invalid_moves)
        if np.sum(output_2d) == 0:
            print("all zero output")
            output_2d += np.logical_not(search_node.invalid_moves)
        return output_2d / np.sum(output_2d), value

    def run_search(self, search_node, player, path):
        # the game ended on this this node
        if search_node.game_has_ended:
            # the reward for the previous player is the opposite of the reward for the next player
            return -search_node.reward

        if search_node.is_unvisited():
            policy, value = self.get_probability_distribution(search_node, player)
            search_node.neural_network_policy = policy
            return -value

        # choose the move having the biggest upper confidence bound
        chosen_move, next_node = self.choose_move(search_node, player)
        search_node.move_forward_selected(chosen_move)
        v = self.run_search(next_node, player.get_other_player(), path)
        search_node.update_expected_value_for_move(chosen_move, v)
        return -v

    def choose_move(self, search_node, player):
        upper_confidence_bounds = search_node.expected_move_rewards + self.exploration_rate * \
                                  search_node.neural_network_policy * np.sqrt(search_node.visited_count + 1e-8) / \
                                  (1 + search_node.forward_visited_counts)
        upper_confidence_bounds[search_node.invalid_moves] = -np.inf

        ranked_moves = np.argsort(upper_confidence_bounds.flatten())
        number_of_valid_moves = len(ranked_moves) - search_node.invalid_move_count
        for index, move in enumerate(reversed(ranked_moves)):
            if index >= number_of_valid_moves:
                break
            move_2d = tuple(np.unravel_index(move, search_node.board_state.get_shape()))
            new_board_state = search_node.get_board_state_after_move(move_2d, player)
            node_to_be_searced = self.get_search_node_of_state(new_board_state, move_2d)
            if not node_to_be_searced.pending_policy_calculation:
                return move_2d, node_to_be_searced
        return None, None

    def format_input(self, game_boards: List[np.ndarray], players=None):
        if players is not None:
            own_symbols = np.array(list(map(lambda player: player.get_symbol(), players)))
        else:
            own_symbols = np.array(1)
        own_symbols = own_symbols.reshape((-1, 1, 1))
        numeric_boards = np.array(game_boards)
        own_pieces = np.array(numeric_boards == own_symbols, dtype='float')
        opponent_pieces = np.array(numeric_boards == -own_symbols, dtype='float')
        numeric_representation = np.stack([own_pieces, opponent_pieces], axis=3)
        return numeric_representation

    def train(self, samples: TrainingSampleCollection):
        print('number of training samples: ' + str(samples.get_length()))
        input = self.format_input(samples.board_states)
        output_policies = np.array(samples.move_probabilities)
        output_values = np.array(samples.rewards)
        return self.model.fit(x=input, y=[output_policies, output_values], epochs=self.training_epochs, shuffle=True,
                              verbose=2, batch_size=64)

    def get_name(self):
        return 'MCTSAgent'
