from typing import List, Dict

import numpy as np

from AmoebaPlayGround.Agents.MCTS.DictMCTSTree import MCTSNode
from AmoebaPlayGround.Agents.NetworkModels import PolicyValueNetwork
from AmoebaPlayGround.Agents.NeuralAgent import NetworkModel, NeuralAgent
from AmoebaPlayGround.Amoeba import AmoebaGame
from AmoebaPlayGround.GameBoard import AmoebaBoard
from AmoebaPlayGround.Training.Logger import Statistics
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

    def get_step(self, games: List[AmoebaGame], player, evaluation=False):
        search_nodes = self.get_root_nodes(None, games, evaluation)
        for i in range(self.search_count):
            for node in search_nodes:
                self.run_search(node, player, set())
        return self.get_move_probabilities_from_nodes(search_nodes, player), Statistics()

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
            search_node.set_policy(policy)
            return -value

        # choose the move having the biggest upper confidence bound
        chosen_move, next_node = self.choose_move(search_node, player)
        search_node.move_forward_selected(chosen_move)
        v = self.run_search(next_node, player.get_other_player(), path)
        search_node.update_expected_value_for_move(chosen_move, v)
        return -v

    def choose_move(self, search_node: MCTSNode, search_tree, player):
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
        output_values = np.array(samples.rewards)
        if validation_dataset is not None:
            validation_input = self.format_input(validation_dataset.board_states)
            validation_output_policies = np.array(validation_dataset.move_probabilities)
            validation_output_values = np.array(validation_dataset.rewards)
            validation_dataset = (validation_input, [validation_output_policies, validation_output_values])
        return self.model_type.train(self.model, input, [output_policies, output_values],
                                     validation_data=validation_dataset, **kwargs)

    def get_name(self):
        return 'MCTSAgent'
