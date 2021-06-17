import numpy as np

from AmoebaPlayGround.GameBoard import Player


class TrainingSampleCollection:
    def __init__(self, board_states=None, move_probabilities=None, rewards=None, max_size=50000):
        if rewards is None:
            rewards = []
        if board_states is None:
            board_states = []
        if move_probabilities is None:
            move_probabilities = []
        self.board_states = board_states[0:min(max_size, len(board_states))]
        self.move_probabilities = move_probabilities[0:min(max_size, len(move_probabilities))]
        self.rewards = rewards[0:min(max_size, len(rewards))]
        self.max_size = max_size

    def get_length(self):
        return len(self.rewards)

    def remove_samples_from_front(self, count):
        self.board_states = self.board_states[count:]
        self.move_probabilities = self.move_probabilities[count:]
        self.rewards = self.rewards[count:]

    def extend(self, training_sample_collection):
        free_space = self.max_size - self.get_length()
        samples_to_remove = max(training_sample_collection.get_length() - free_space, 0)
        self.remove_samples_from_front(samples_to_remove)
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
