from typing import List

import numpy as np

from AmoebaPlayGround.Agents.MCTS.BaseMCTSTree import MCTSNode
from AmoebaPlayGround.Agents.MCTS.DictMCTSTree import DictMCTSTree
from AmoebaPlayGround.Agents.MCTS.MCTSAgent import MCTSAgent
from AmoebaPlayGround.Agents.TensorflowModels import NetworkModel, ResNetLike
from AmoebaPlayGround.Amoeba import AmoebaGame
from AmoebaPlayGround.Training.Logger import Statistics


class PositionToSearch:
    def __init__(self, search_node: MCTSNode, search_tree: dict, searches_remaining: int, id: int):
        self.search_node = search_node
        self.search_tree = search_tree
        self.searches_remaining = searches_remaining
        self.parallel_searches = 0
        self.id = id

    def selected_into_batch(self):
        self.searches_remaining -= 1
        self.parallel_searches += 1

    def search_ended(self):
        self.searches_remaining -= 1

    def finished_search(self):
        return self.searches_remaining < 1

    def new_batch_started(self):
        self.parallel_searches = 0


class BatchMCTSAgent(MCTSAgent):
    def __init__(self, model_name=None, load_latest_model=False,
                 model_type: NetworkModel = ResNetLike(6), search_count=100, exploration_rate=6.0,
                 batch_size=20, training_epochs=4, dirichlet_ratio=0.25, map_size=(8, 8),
                 tree_type=DictMCTSTree, max_intra_game_parallelism=16, neural_network_evaluator=None,
                 virtual_loss=3, training_dataset_max_size=200000):
        super().__init__(model_name, load_latest_model, model_type, search_count, exploration_rate, training_epochs,
                         dirichlet_ratio, map_size, training_dataset_max_size)
        self.batch_size = batch_size
        self.statistics = Statistics()
        self.search_trees = dict()
        self.tree_type = tree_type
        self.virtual_loss = virtual_loss
        self.max_intra_game_parallelism = max_intra_game_parallelism
        self.config = {"model_name": model_name, "load_latest_model": load_latest_model, "model_type": model_type,
                       "search_count": search_count, "exploration_rate": exploration_rate, "batch_size": batch_size,
                       "training_epochs": training_epochs, "dirichlet_ratio": dirichlet_ratio, "map_size": map_size,
                       "tree_type": tree_type, "max_intra_game_parallelism": max_intra_game_parallelism,
                       "virtual_loss": virtual_loss, "training_dataset_max_size": training_dataset_max_size}

    def reset_statistics(self):
        self.statistics = Statistics()

    def get_config(self):
        return self.config

    def get_copy(self):
        new_instance = self.__class__(model_type=self.model_type, search_count=self.search_count,
                                      exploration_rate=self.exploration_rate, training_epochs=self.training_epochs,
                                      dirichlet_ratio=self.dirichlet_ratio, tree_type=self.tree_type,
                                      batch_size=self.batch_size, map_size=self.map_size,
                                      max_intra_game_parallelism=self.max_intra_game_parallelism,
                                      virtual_loss=self.virtual_loss,
                                      training_dataset_max_size=self.training_dataset_max_size)
        new_instance.set_weights(self.get_weights())
        return new_instance

    def get_step(self, games: List[AmoebaGame], player, evaluation=False):
        search_trees = self.get_search_trees_for_games(games)
        self.reset_statistics()
        positions_to_search, finished_positions = self.get_positions_to_search(search_trees, games, evaluation)

        while len(positions_to_search) > 0:
            paths, leaf_nodes, last_players = self.run_selection(positions_to_search, player)
            if len(leaf_nodes) > 0:
                policies, values = self.run_simulation(leaf_nodes, last_players)
                self.set_policies(leaf_nodes, policies, paths)
                self.run_back_propagation(paths, values, self.virtual_loss)
            positions_to_search = self.move_over_fully_searched_games(positions_to_search, finished_positions)

        self.statistics.add_tree_statistics(finished_positions)
        return self.get_move_probabilities_from_nodes(list(map(lambda p: p.search_node, finished_positions)),
                                                      player), self.statistics


    def get_search_trees_for_games(self, games):
        updated_tree_dictionary = dict()
        trees = []
        for game in games:
            stored_tree = self.search_trees.get(game.id)
            if stored_tree is not None:
                updated_tree_dictionary[game.id] = stored_tree
                trees.append(stored_tree)
            else:
                new_tree = self.tree_type(game.num_steps)
                updated_tree_dictionary[game.id] = new_tree
                trees.append(new_tree)
        self.search_trees = updated_tree_dictionary
        return trees

    def move_over_fully_searched_games(self, positions_to_search, finished_positions):
        remaining_postions_to_search = []
        for position_to_search in positions_to_search:
            if position_to_search.finished_search():
                finished_positions[position_to_search.id] = position_to_search
            else:
                remaining_postions_to_search.append(position_to_search)
                position_to_search.new_batch_started()
        return remaining_postions_to_search

    def get_positions_to_search(self, search_trees, games, evaluation):
        search_nodes = self.get_root_nodes(search_trees, games, evaluation)
        positions_to_search = []
        finished_placeholders = []
        id_counter = 0
        for node, search_tree in zip(search_nodes, search_trees):
            positions_to_search.append(PositionToSearch(node, search_tree, self.search_count, id_counter))
            id_counter += 1
            finished_placeholders.append(None)
        return positions_to_search, finished_placeholders

    def replace_values_in_game_ending_nodes(self, leaf_nodes, values):
        corrected_values = []
        for leaf_node, value in zip(leaf_nodes, values):
            if leaf_node.has_game_ended():
                corrected_values.append(leaf_node.reward)
            else:
                corrected_values.append(value)
        return corrected_values

    def set_policies(self, nodes, policies, paths):
        for node, policy in zip(nodes, policies):
            node.set_policy(policy)
            node.policy_calculation_ended()

    def run_selection(self, positions_to_search, player):
        paths = []
        leaf_nodes = []
        last_players = []
        positions_to_search = sorted(positions_to_search, key=lambda x: x.searches_remaining, reverse=True)
        while len(paths) < self.batch_size and len(positions_to_search) > 0:
            remaining_positions_to_search = []
            for position in positions_to_search:
                path, end_node, end_player, can_continue_search = self.run_selection_for_node(position, player)
                if path is not None:
                    paths.append(path)
                    leaf_nodes.append(end_node)
                    last_players.append(end_player)
                    if len(paths) >= self.batch_size:
                        break
                if can_continue_search:
                    remaining_positions_to_search.append(position)

            positions_to_search = remaining_positions_to_search

        return paths, leaf_nodes, last_players

    def run_selection_for_node(self, position: PositionToSearch, player):
        current_node: MCTSNode = position.search_node

        if position.parallel_searches > self.max_intra_game_parallelism:
            return None, None, None, False

        current_player = player
        path = []

        if current_node.is_unvisited():
            return path, current_node, current_player, False

        while True:
            if current_node.has_game_ended():
                self.run_back_propagation([path], [current_node.reward], self.virtual_loss)
                position.search_ended()
                if 0 >= position.searches_remaining:
                    return None, None, None, False
                path = []
                current_node = position.search_node
                current_player = player
            if current_node.is_unvisited():
                position.selected_into_batch()
                self.node_selected(current_node, path)
                return path, current_node, current_player, True

            chosen_move, next_node = self.choose_move_vectorized(current_node, position.search_tree, current_player)
            if chosen_move is None:
                self.selection_cancelled(path)
                return None, None, None, False
            path.append((current_node, chosen_move))
            current_node.node_selected(chosen_move, self.virtual_loss)

            current_node = next_node
            current_player = current_player.get_other_player()

    def node_selected(self, node, path):
        pass

    def selection_cancelled(self, path):
        for node, move in path:
            node.selection_cancelled(move, self.virtual_loss)

    def run_simulation(self, leaf_nodes: List[MCTSNode], players):
        board_states = list(map(lambda node: node.board_state.cells, leaf_nodes))
        board_size = board_states[0].shape
        input = self.format_input(board_states, players)
        invalid_moves = list(map(lambda node: node.invalid_moves, leaf_nodes))
        invalid_moves = np.array(invalid_moves)

        output_2d, value = self.model.predict(input, batch_size=self.batch_size)
        output_2d = output_2d.reshape(-1, board_size[0], board_size[1])
        valid_moves = np.logical_not(invalid_moves)
        output_2d = output_2d * valid_moves
        # handle all zero outputs
        output_sum = np.sum(output_2d, axis=(1, 2))
        zero_outputs = output_sum == 0
        if np.sum(zero_outputs) > 0:
            print("{count} outputs are all zero in batch".format(count=np.sum(zero_outputs)))
            output_2d = np.where(zero_outputs.reshape((-1, 1, 1)), valid_moves, output_2d)
            output_sum = np.sum(output_2d, axis=(1, 2))

        return output_2d / output_sum.reshape((-1, 1, 1)), value

    def run_back_propagation(self, paths, values, virtual_loss_to_remove=0):
        # values are received from the perspective of the leaf node and paths does not contain them
        values = -np.array(values)

        for path, value in zip(paths, values):
            for node, move in reversed(path):
                node.update_expected_value_for_move(move, value, virtual_loss_to_remove)
                value = -value
        self.statistics.add_searches(paths)

    def get_name(self):
        return 'BatchMCTSAgent'