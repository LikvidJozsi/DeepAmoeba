from abc import ABC
from typing import Dict

import numpy as np
from numba import njit

from AmoebaPlayGround.Agents.AmoebaAgent import AmoebaAgent
from AmoebaPlayGround.Agents.MCTS.DictMCTSTree import MCTSNode
from AmoebaPlayGround.Agents.TensorflowModels import NeuralNetworkModel
from AmoebaPlayGround.GameBoard import AmoebaBoard
from AmoebaPlayGround.Training.TrainingSampleGenerator import TrainingSampleCollection, TrainingDatasetGenerator


class MCTSAgent(AmoebaAgent, ABC):

    def __init__(self, model: NeuralNetworkModel,
                 search_count=100, exploration_rate=1.4,
                 training_epochs=10, dirichlet_ratio=0.25, map_size=(8, 8), training_dataset_max_size=200000):
        self.model = model
        self.map_size = map_size
        self.mcts_nodes: Dict[AmoebaBoard, MCTSNode] = {}
        self.search_count = search_count
        self.exploration_rate = exploration_rate
        self.training_epochs = training_epochs
        self.dirichlet_ratio = dirichlet_ratio
        self.evaluation = False
        self.training_dataset_max_size = training_dataset_max_size

    def set_training_mode(self):
        self.evaluation = False

    def set_evaluation_mode(self):
        self.evaluation = True

    def get_neural_network_model(self):
        return self.model

    def get_copy(self):
        print("this is not getting called right?")
        new_instance = self.__class__(model=self.model.get_copy(), search_count=self.search_count,
                                      exploration_rate=self.exploration_rate, training_epochs=self.training_epochs,
                                      dirichlet_ratio=self.dirichlet_ratio,
                                      training_dataset_max_size=self.training_dataset_max_size)
        return new_instance

    def get_root_nodes(self, search_trees, games):
        nodes = []
        if self.evaluation:
            eps = 0
        else:
            eps = self.dirichlet_ratio

        for game, search_tree in zip(games, search_trees):
            root_node = search_tree.get_root_node(game, eps)
            nodes.append(root_node)
            search_tree.set_turn(game.num_steps + 1)
        return nodes

    def get_move_probabilities_from_nodes(self, nodes, player):
        action_probabilities = []
        for node in nodes:
            action_visited_counts = node.move_visited_counts
            probabilities = action_visited_counts / np.sum(action_visited_counts)
            action_probabilities.append(probabilities)
        return action_probabilities

    def choose_move_vectorized(self, search_node: MCTSNode, search_tree, player):
        best_move = get_best_ucb_node(search_node.sum_expected_move_rewards,
                                      search_node.get_policy(), search_node.move_visited_counts,
                                      search_node.invalid_moves, self.exploration_rate, search_node.visited_count,
                                      search_node.board_state.get_shape())
        if best_move is None:
            return None, None

        chosen_move_node = search_tree.get_the_node_of_move(search_node, best_move, player)
        return best_move, chosen_move_node

    def choose_move_non_vectorized(self, search_node: MCTSNode, search_tree, player):
        average_expected_reward = search_node.sum_expected_move_rewards / np.maximum(1, search_node.move_visited_counts)
        upper_confidence_bounds = average_expected_reward + self.exploration_rate * search_node.neural_network_policy * \
                                  np.sqrt(search_node.visited_count + 1e-8) / (1 + search_node.move_visited_counts)
        upper_confidence_bounds[search_node.invalid_moves] = -np.inf

        valid_upper_confidence_bounds = np.where(search_node.invalid_moves, -np.inf, upper_confidence_bounds)
        best_move = np.argmax(valid_upper_confidence_bounds)
        board_shape = search_node.board_state.get_shape()
        index_1 = best_move // board_shape[0]
        index_2 = best_move - index_1 * board_shape[0]
        best_move = (index_1, index_2)
        if valid_upper_confidence_bounds[best_move] == -np.inf:
            return None

        chosen_move_node = search_tree.get_the_node_of_move(search_node, best_move, player)
        return best_move, chosen_move_node

    def train(self, dataset_generator: TrainingDatasetGenerator,
              validation_dataset: TrainingSampleCollection = None):
        print('number of training samples: ' + str(dataset_generator.get_sample_count()))
        inputs, output_policies, output_values = dataset_generator.get_dataset(self.training_dataset_max_size)
        output_policies = output_policies.reshape(output_policies.shape[0], -1)
        return self.model.train(inputs, output_policies, output_values, validation_dataset=validation_dataset)

    def get_name(self):
        return 'MCTSAgent'


@njit(fastmath=True)
def get_best_ucb_node(sum_expected_move_rewards, neural_network_policy, move_visited_counts,
                      invalid_moves, exploration_rate, visited_count, board_shape):
    average_expected_reward = sum_expected_move_rewards / np.maximum(1, move_visited_counts)
    upper_confidence_bounds = average_expected_reward + exploration_rate * neural_network_policy * \
                              np.sqrt(visited_count + 1e-8) / (1 + move_visited_counts)

    valid_upper_confidence_bounds = np.where(invalid_moves, -np.inf, upper_confidence_bounds)
    best_move = np.argmax(valid_upper_confidence_bounds)
    index_1 = best_move // board_shape[0]
    index_2 = best_move - index_1 * board_shape[0]
    best_move = (index_1, index_2)
    if valid_upper_confidence_bounds[best_move] == -np.inf:
        return None
    return best_move
