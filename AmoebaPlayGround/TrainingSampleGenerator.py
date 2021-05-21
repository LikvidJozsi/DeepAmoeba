import numpy as np

from AmoebaPlayGround.GameBoard import Player


class TrainingSampleCollection:
    def __init__(self, board_states=None, move_probabilities=None, rewards=None):
        if rewards is None:
            rewards = []
        if board_states is None:
            board_states = []
        if move_probabilities is None:
            move_probabilities = []
        self.board_states = board_states
        self.move_probabilities = move_probabilities
        self.rewards = rewards

    def get_length(self):
        return len(self.rewards)

    def extend(self, training_sample_collection):
        self.board_states.extend(training_sample_collection.board_states)
        self.move_probabilities.extend(training_sample_collection.move_probabilities)
        self.rewards.extend(training_sample_collection.rewards)


class TrainingSampleGenerator():
    def get_training_data(self, winner):
        pass

    def add_move(self, board_state, probabilities, player):
        pass


class SymmetricTrainingSampleGenerator(TrainingSampleGenerator):
    def __init__(self):
        self.board_states = []
        self.move_probabilities = []
        self.players = []

    def add_move(self, board_state, probabilities, player):
        for i in range(4):
            self.board_states.append(board_state)
            self.move_probabilities.append(probabilities)
            self.players.append(player)
            board_state = np.rot90(board_state)
            probabilities = np.rot90(probabilities)

    def get_training_data(self, winner):
        rewards = list(
            map(lambda player: 1 if player == winner else 0 if winner == Player.NOBODY else -1, self.players))
        return TrainingSampleCollection(self.board_states, self.move_probabilities, rewards)
