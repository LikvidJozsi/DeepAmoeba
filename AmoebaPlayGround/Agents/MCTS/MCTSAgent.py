from typing import List, Dict

import numpy as np
from numba import njit

from AmoebaPlayGround.Agents.MCTS.DictMCTSTree import MCTSNode
from AmoebaPlayGround.Agents.NetworkModels import PolicyValueNetwork
from AmoebaPlayGround.Agents.NeuralAgent import NetworkModel, NeuralAgent
from AmoebaPlayGround.GameBoard import AmoebaBoard
from AmoebaPlayGround.Training.TrainingSampleGenerator import TrainingSampleCollection


class MCTSAgent(NeuralAgent):

    def __init__(self, model_name=None, load_latest_model=False,
                 model_type: NetworkModel = PolicyValueNetwork(), search_count=100, exploration_rate=1.4,
                 training_epochs=10, dirichlet_ratio=0.25, map_size=(8, 8)):
        super().__init__(model_type, model_name, load_latest_model, map_size)
        self.mcts_nodes: Dict[AmoebaBoard, MCTSNode] = {}
        self.search_count = search_count
        self.exploration_rate = exploration_rate
        self.training_epochs = training_epochs
        self.dirichlet_ratio = dirichlet_ratio

    def get_copy(self):
        print("this is not getting called right?")
        new_instance = self.__class__(model_type=self.model_type, search_count=self.search_count,
                                      exploration_rate=self.exploration_rate, training_epochs=self.training_epochs,
                                      dirichlet_ratio=self.dirichlet_ratio)
        new_instance.set_weights(self.get_weights())
        return new_instance

    def get_root_nodes(self, search_trees, games, evaluation):
        nodes = []
        if evaluation:
            eps = 0
        else:
            eps = self.dirichlet_ratio

        for game, search_tree in zip(games, search_trees):
            root_node = search_tree.get_root_node(game, eps)
            nodes.append(root_node)

        return nodes

    def get_move_probabilities_from_nodes(self, nodes, player):
        action_probabilities = []
        for node in nodes:
            action_visited_counts = node.forward_visited_counts
            probabilities = action_visited_counts / np.sum(action_visited_counts)
            action_probabilities.append(probabilities)
        return action_probabilities

    def choose_move_vectorized(self, search_node: MCTSNode, search_tree, player):
        best_move = get_best_ucb_node(search_node.backward_visited_counts, search_node.sum_expected_move_rewards,
                                      search_node.get_policy(), search_node.forward_visited_counts,
                                      search_node.invalid_moves, self.exploration_rate, search_node.visited_count,
                                      search_node.board_state.get_shape())
        if best_move is None:
            return None, None

        chosen_move_node = search_tree.get_the_node_of_move(search_node, best_move, player)
        return best_move, chosen_move_node

    def choose_move_non_vectorized(self, search_node: MCTSNode, search_tree, player):
        average_expected_reward = np.where(search_node.backward_visited_counts == 0, 0,
                                           search_node.sum_expected_move_rewards / search_node.backward_visited_counts)
        upper_confidence_bounds = average_expected_reward + self.exploration_rate * search_node.neural_network_policy * \
                                  np.sqrt(search_node.visited_count + 1e-8) / (1 + search_node.forward_visited_counts)
        upper_confidence_bounds[search_node.invalid_moves] = -np.inf

        ranked_moves = np.argsort(upper_confidence_bounds.flatten())
        number_of_valid_moves = len(ranked_moves) - search_node.invalid_move_count
        for index, move in enumerate(reversed(ranked_moves)):
            if index >= number_of_valid_moves:
                break
            move_2d = tuple(np.unravel_index(move, search_node.board_state.get_shape()))
            chosen_move_node = search_tree.get_the_node_of_move(search_node, move_2d, player)
            if not chosen_move_node.pending_policy_calculation:
                return move_2d, chosen_move_node
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

    def train(self, samples: TrainingSampleCollection,
              validation_dataset: TrainingSampleCollection = None,
              **kwargs):
        print('number of training samples: ' + str(samples.get_length()))
        input = self.format_input(samples.board_states)
        output_policies = np.array(samples.move_probabilities)
        output_policies = output_policies.reshape(output_policies.shape[0], -1)
        output_values = np.array(samples.rewards)
        if validation_dataset is not None:
            validation_input = self.format_input(validation_dataset.board_states)
            validation_output_policies = np.array(validation_dataset.move_probabilities)
            validation_output_policies = validation_output_policies.reshape(validation_output_policies.shape[0], -1)
            validation_output_values = np.array(validation_dataset.rewards)
            validation_dataset = (validation_input, [validation_output_policies, validation_output_values])
        return self.model_type.train(self.model, input, [output_policies, output_values],
                                     validation_data=validation_dataset, **kwargs)

    def get_name(self):
        return 'MCTSAgent'


@njit(fastmath=True)
def get_best_ucb_node(backward_visited_counts, sum_expected_move_rewards, neural_network_policy, forward_visited_counts,
                      invalid_moves, exploration_rate, visited_count, board_shape):
    average_expected_reward = np.where(backward_visited_counts == 0, 0,
                                       sum_expected_move_rewards / backward_visited_counts)
    upper_confidence_bounds = average_expected_reward + exploration_rate * neural_network_policy * \
                              np.sqrt(visited_count + 1e-8) / (1 + forward_visited_counts)

    valid_upper_confidence_bounds = np.where(invalid_moves, -np.inf, upper_confidence_bounds)
    best_move = np.argmax(valid_upper_confidence_bounds)
    index_1 = best_move // board_shape[0]
    index_2 = best_move - index_1 * board_shape[0]
    best_move = (index_1, index_2)
    if valid_upper_confidence_bounds[best_move] == -np.inf:
        return None
    return best_move
