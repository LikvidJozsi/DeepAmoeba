import copy
from typing import List, Set

import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.optimizer_v1 import Adam

from AmoebaPlayGround.Amoeba import AmoebaGame
from AmoebaPlayGround.GameBoard import AmoebaBoard, Symbol, Player
from AmoebaPlayGround.NeuralAgent import NetworkModel, NeuralAgent
from AmoebaPlayGround.TrainingSampleGenerator import TrainingSampleCollection


class PolicyValueNetwork(NetworkModel):
    def __init__(self, first_convolution_size=(9, 9), dropout=0.5):
        self.first_convolution_size = first_convolution_size
        self.dropout = dropout

    def create_model(self, map_size):
        input = Input(shape=map_size + (2,))
        conv_1 = Activation('relu')(
            BatchNormalization(axis=3)(Conv2D(32, kernel_size=self.first_convolution_size, padding='same')(input)))
        conv_2 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(conv_1)))
        conv_3 = Activation('relu')(
            BatchNormalization(axis=3)(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(conv_2)))
        flatten = Flatten()(conv_3)
        dense_1 = Dropout(self.dropout)(Activation('relu')(Dense(512, activation='relu')(flatten)))
        dense_2 = Dropout(self.dropout)(Activation('relu')(Dense(128, activation='relu')(dense_1)))
        policy = Dense(np.prod(map_size), activation='softmax')(dense_2)
        value = Dense(1, activation='tanh')(dense_2)

        model = Model(inputs=input, outputs=[policy, value])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.3))
        return model


class MCTSNode:
    def __init__(self, board_state: AmoebaBoard, player: Player, has_game_ended=False):
        self.board_state: AmoebaBoard = board_state
        self.expected_move_rewards: np.ndarray[np.float32] = np.zeros(board_state.get_shape(), dtype=np.float32)
        self.move_visited_counts: np.ndarray[np.int32] = np.zeros(board_state.get_shape(), dtype=np.int32)
        self.valid_moves = np.array(board_state.cells == Symbol.EMPTY, dtype=np.uint8)
        self.visited_count = 0
        self.neural_network_policy = None
        self.game_has_ended = has_game_ended
        self.next_player = player
        self.reward = None

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

    def __eq__(self, other):
        return self.board_state == other.board_state

    def get_node_for_move(self, move, player):
        new_board_state = copy.deepcopy(self.board_state)
        new_board_state.set(move, player.get_symbol())
        new_node = MCTSNode(new_board_state, player.get_other_player())
        new_node.set_game_ended(move)
        return new_node

    def update_expected_value_for_move(self, move, simulation_value):
        visited_count = self.move_visited_counts[move]
        expected_reward = self.expected_move_rewards[move]
        self.expected_move_rewards[move] = (visited_count * expected_reward + simulation_value) / (
                visited_count + 1)
        self.move_visited_counts[move] += 1


class MCTSAgent(NeuralAgent):

    def __init__(self, model_name=None, load_latest_model=False,
                 model_type: NetworkModel = PolicyValueNetwork(), simulation_count=100, exploration_rate=1.4):
        super().__init__(model_name, load_latest_model, model_type)
        self.mcts_nodes: Set[MCTSNode] = {}
        self.simulation_count = simulation_count
        self.exploration_rate = exploration_rate

    def get_step(self, game_boards: List[AmoebaBoard], player):
        for i in range(self.simulation_count):
            for game_board in game_boards:
                search_node = self.get_search_node_of_state(game_board)
                self.runSimulation(search_node, player)

    def get_search_node_of_state(self, board_state):
        searching_node = MCTSNode(board_state, False)
        if searching_node in self.mcts_nodes:
            return self.mcts_nodes[searching_node]
        else:
            self.mcts_nodes.add(searching_node)
            return searching_node

    def get_probability_distribution(self, game_board, player):
        formatted_input = self.format_input(game_board, player)
        output_1d, value = self.model.predict(formatted_input, batch_size=256)
        output_2d = output_1d.rehspae(game_board.shape)
        valid_moves = np.array(game_board == Symbol.EMPTY)
        output_2d = output_2d * valid_moves
        if np.sum(output_2d) == 0:
            output_2d += valid_moves
        return output_2d / np.sum(output_2d), value, valid_moves

    def runSimulation(self, search_node, player):
        # the game ended on this this node
        if search_node.game_has_ended:
            # the reward for the previous player is the opposite of the reward for the next player
            return -search_node.reward

        # we have not visited this node yet
        if search_node.neural_network_policy is None:
            policy, value, valid_moves = self.get_probability_distribution(search_node.board_state, player)
            search_node.neural_network_policy = policy
            search_node.valid_moves = valid_moves
            return -value

        # choose the move having the biggest upper confidence bound
        upper_confidence_bounds = search_node.expected_move_rewards + self.exploration_rate * \
                                  search_node.neural_network_policy * np.sqrt(search_node.visited_count + 1e-8) + \
                                  (1 + search_node.move_visited_counts)
        chosen_move = np.argmax(upper_confidence_bounds)
        chosen_move_2d = np.unravel_index(chosen_move, search_node.board_state.get_shape())
        next_node = search_node.get_node_for_move(chosen_move_2d, player)
        v = self.runSimulation(next_node)
        search_node.update_expected_value_for_move(chosen_move_2d, v)
        search_node.visited_count += 1
        return -v

    def format_input(self, game_boards: List[np.ndarray]):
        numeric_boards = np.array(game_boards)
        # training samples are are already generated so 1 is always the player pieces and -1 the opponents
        own_pieces = np.array(numeric_boards == 1, dtype='float')
        opponent_pieces = np.array(numeric_boards == -1, dtype='float')
        numeric_representation = np.stack([own_pieces, opponent_pieces], axis=3)
        return numeric_representation

    def train(self, training_sample_collection: TrainingSampleCollection):
        print('number of training samples: ' + str(training_sample_collection.get_length()))
        input = self.format_input(training_sample_collection.board_states)
        output_policies = np.array(training_sample_collection.move_probabilities)
        output_values = np.array(training_sample_collection.rewards)
        return self.model.fit(x=input, y=(output_policies, output_values), epochs=15, shuffle=True, verbose=2,
                              batch_size=32)
