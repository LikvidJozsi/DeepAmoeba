import copy
from typing import List, Dict

import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dropout

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
        dense_1 = Dropout(self.dropout)(Activation('relu')(Dense(256, activation='relu')(flatten)))
        dense_2 = Dropout(self.dropout)(Activation('relu')(Dense(128, activation='relu')(dense_1)))
        policy = Dense(np.prod(map_size), activation='softmax')(dense_1)
        value = Dense(1, activation='tanh')(dense_2)

        model = Model(inputs=input, outputs=[policy, value])
        model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(lr=0.3))
        return model


class MCTSNode:
    def __init__(self, board_state: AmoebaBoard, has_game_ended=False):
        self.board_state: AmoebaBoard = board_state
        self.expected_move_rewards: np.ndarray[np.float32] = np.zeros(board_state.get_shape(), dtype=np.float32)
        self.move_visited_counts: np.ndarray[np.uint16] = np.zeros(board_state.get_shape(), dtype=np.uint16)
        self.invalid_moves = board_state.cells != Symbol.EMPTY.value
        self.visited_count = 0
        self.neural_network_policy = None
        self.game_has_ended = has_game_ended
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

    def get_board_state_for_move(self, move, player):
        new_board_state = self.board_state.copy()
        new_board_state.set(move, player.get_symbol())
        return new_board_state

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
        self.mcts_nodes: Dict[AmoebaBoard, MCTSNode] = {}
        self.simulation_count = simulation_count
        self.exploration_rate = exploration_rate

    def reset(self):
        self.mcts_nodes = {}

    def get_step(self, game_boards: List[AmoebaBoard], player):
        for i in range(self.simulation_count):
            for game_board in game_boards:
                board_copy = game_board.copy()
                search_node = self.get_search_node_of_state(board_copy)
                self.runSimulation(search_node, player,0)
        action_probabilities = []
        for game_board in game_boards:
            search_node = self.get_search_node_of_state(game_board)
            action_visited_counts = search_node.move_visited_counts
            probabilities = action_visited_counts / np.sum(action_visited_counts)
            action_probabilities.append(probabilities)
        return action_probabilities

    def get_search_node_of_state(self, board_state, move=None):
        if board_state in self.mcts_nodes:
            return self.mcts_nodes[board_state]
        else:
            new_node = MCTSNode(board_state)
            if move is not None:
                new_node.set_game_ended(move)
            self.mcts_nodes[board_state] = new_node
            return new_node

    def get_probability_distribution(self, search_node, player):
        game_board = search_node.board_state.cells
        formatted_input = self.format_input([game_board], player)
        output_1d, value = self.model.predict(formatted_input, batch_size=256)
        output_2d = output_1d.reshape(game_board.shape)

        output_2d = output_2d * np.logical_not(search_node.invalid_moves)
        if np.sum(output_2d) == 0:
            print("all zero output")
            output_2d += search_node.valid_moves
        return output_2d / np.sum(output_2d), value

    def runSimulation(self, search_node, player,depth):
        # the game ended on this this node
        if search_node.game_has_ended:
            # the reward for the previous player is the opposite of the reward for the next player
            return -search_node.reward

        # we have not visited this node yet
        if search_node.neural_network_policy is None:
            policy, value = self.get_probability_distribution(search_node,player)
            search_node.neural_network_policy = policy
            return -value

        # choose the move having the biggest upper confidence bound
        upper_confidence_bounds = search_node.expected_move_rewards + self.exploration_rate * \
                                  search_node.neural_network_policy * np.sqrt(search_node.visited_count + 1e-8) / \
                                  (1 + search_node.move_visited_counts)
        upper_confidence_bounds[search_node.invalid_moves] = -np.inf
        chosen_move = np.argmax(upper_confidence_bounds)
        chosen_move_2d = tuple(np.unravel_index(chosen_move, search_node.board_state.get_shape()))
        new_board_state = search_node.get_board_state_for_move(chosen_move_2d, player)
        next_node = self.get_search_node_of_state(new_board_state, chosen_move_2d)
        v = self.runSimulation(next_node, player.get_other_player(),depth+1)
        search_node.update_expected_value_for_move(chosen_move_2d, v)
        search_node.visited_count += 1
        return -v

    def format_input(self, game_boards: List[np.ndarray], player=None):
        if player is not None:
            own_symbol = player.get_symbol().value
        else:
            own_symbol = 1
        numeric_boards = np.array(game_boards)
        own_pieces = np.array(numeric_boards == own_symbol, dtype='float')
        opponent_pieces = np.array(numeric_boards == -own_symbol, dtype='float')
        numeric_representation = np.stack([own_pieces, opponent_pieces], axis=3)
        return numeric_representation

    def train(self, samples: TrainingSampleCollection):
        print('number of training samples: ' + str(samples.get_length()))
        input = self.format_input(samples.board_states)
        output_policies = np.array(samples.move_probabilities).reshape((samples.get_length(), -1))
        output_values = np.array(samples.rewards)
        return self.model.fit(x=input, y=[output_policies, output_values], epochs=30, shuffle=True, verbose=2,
                              batch_size=32)

    def get_name(self):
        return 'MCTSAgent'
