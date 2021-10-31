import numpy as np

from AmoebaPlayGround.GameBoard import Player


class TrainingSampleCollection:
    def __init__(self, board_states=None, move_probabilities=None, rewards=None, max_size=None):
        if rewards is None:
            rewards = []
        if board_states is None:
            board_states = []
        if move_probabilities is None:
            move_probabilities = []
        if max_size is not None:
            board_states = board_states[0:min(max_size, len(board_states))]
            move_probabilities = move_probabilities[0:min(max_size, len(move_probabilities))]
            rewards = rewards[0:min(max_size, len(rewards))]
        self.board_states = board_states
        self.move_probabilities = move_probabilities
        self.rewards = rewards
        self.max_size = max_size

    def get_length(self):
        return len(self.rewards)

    def remove_samples_from_front(self, count):
        self.board_states = self.board_states[count:]
        self.move_probabilities = self.move_probabilities[count:]
        self.rewards = self.rewards[count:]

    def extend(self, training_sample_collection):
        if self.max_size is not None:
            free_space = self.max_size - self.get_length()
            samples_to_remove = max(training_sample_collection.get_length() - free_space, 0)
            self.remove_samples_from_front(samples_to_remove)
        self.board_states.extend(training_sample_collection.board_states)
        self.move_probabilities.extend(training_sample_collection.move_probabilities)
        self.rewards.extend(training_sample_collection.rewards)

    def add_sample(self, board_state, move_probability, reward):
        self.board_states.append(board_state)
        self.move_probabilities.append(move_probability)
        self.rewards.append(reward)

    def add_mirror_symmetries(self, board_state, probabilities, reward):
        self.add_rotations(board_state, probabilities, reward)
        board_state = np.flip(board_state, 0)
        probabilities = np.flip(probabilities, 0)
        self.add_rotations(board_state, probabilities, reward)

    def add_rotations(self, board_state, probabilities, reward):

        for i in range(4):
            self.board_states.append(board_state)
            self.move_probabilities.append(probabilities)
            self.rewards.append(reward)
            board_state = np.rot90(board_state)
            probabilities = np.rot90(probabilities)

    def create_rotational_variations(self, entropy_cutoff=float("inf")):
        original_board_states = self.board_states
        original_move_probabilities = self.move_probabilities
        original_rewards = self.rewards
        self.board_states = []
        self.move_probabilities = []
        self.rewards = []

        for board_state, probability_map, reward in zip(original_board_states, original_move_probabilities,
                                                        original_rewards):
            entropy = self.calculate_entropy(probability_map)
            if entropy < entropy_cutoff:  # TODO board size independent value
                self.add_mirror_symmetries(board_state, probability_map, reward)
            # elif random.random() < 0.1: # there is a 10% chance to include it even if it doesnt meet criteria
            #    self.add_mirror_symmetries(board_state, probabilities, player)

    def calculate_entropy(self, probabilites):
        probabilites = np.maximum(probabilites.flatten(), 1e-8)
        entropy = - probabilites.dot(np.log2(probabilites))
        return entropy

    def print(self):
        for board_state, probabilities, reward in zip(self.board_states, self.move_probabilities, self.rewards):
            print(board_state)
            print(probabilities)
            print(reward)


class PlaceholderTrainingSampleGenerator:
    def get_training_data(self, winner):
        return TrainingSampleCollection()

    def add_move(self, board_state, probabilities, player):
        pass


class SymmetricTrainingSampleGenerator(PlaceholderTrainingSampleGenerator):
    def __init__(self):

        self.board_states = []
        self.move_probabilities = []
        self.players = []

    def add_move(self, board_state, probabilities, player):
        self.board_states.append(board_state)
        self.move_probabilities.append(probabilities)
        self.players.append(player)

    def get_training_data(self, winner):
        rewards = list(
            map(lambda player: 1 if player == winner else 0 if winner == Player.NOBODY else -1, self.players))
        return TrainingSampleCollection(self.board_states, self.move_probabilities, rewards)
