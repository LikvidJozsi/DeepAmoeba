import sys

import numpy as np
from numpy.random import default_rng

from AmoebaPlayGround.GameBoard import Player


class TrainingDatasetGenerator:
    def __init__(self, training_sample_collection=None, episode_window_width=6):
        if training_sample_collection is not None:
            self.sample_collections = [training_sample_collection]
        else:
            self.sample_collections = []
        self.episode_window_width = episode_window_width
        self.symmetry_count = 8
        self.cumulative_dataset_sizes = []

    def add_episode(self, collection):
        if len(self.sample_collections) == self.episode_window_width:
            self.sample_collections.pop(0)
        self.sample_collections.append(collection)

    def set_episode_window_width(self, width):
        if len(self.sample_collections) > width:
            self.sample_collections = self.sample_collections[(len(self.sample_collections) - width):]
        self.episode_window_width = width

    def get_sample_count(self):
        self.update_dataset_sizes()
        return self.cumulative_dataset_sizes[-1] * self.symmetry_count

    def update_dataset_sizes(self):
        self.cumulative_dataset_sizes = np.zeros(len(self.sample_collections), dtype=np.int32)
        cumulative_size = 0
        for index, collection in enumerate(self.sample_collections):
            collection_size = collection.get_length()
            cumulative_size += collection_size
            self.cumulative_dataset_sizes[index] = cumulative_size

    def get_dataset(self, desired_dataset_size=200000):
        self.update_dataset_sizes()
        sample_count = self.get_sample_count()
        dataset_size = min(sample_count, desired_dataset_size)
        rng = default_rng()
        sample_indexes = rng.choice(sample_count, size=dataset_size, replace=False)

        board_size = self.sample_collections[0].get_board_size()
        dataset_board_states = np.empty((dataset_size,) + board_size + (2,), dtype=np.float32)
        dataset_move_probabilities = np.empty((dataset_size,) + board_size, dtype=np.float32)
        dataset_rewards = np.empty((dataset_size,), dtype=np.float32)

        for dataset_index, sample_index in enumerate(sample_indexes):
            (board_state, probabilities, reward, _) = self.get_sample(sample_index)
            dataset_board_states[dataset_index, :, :, 0] = np.array(board_state == 1, dtype=np.float32)
            dataset_board_states[dataset_index, :, :, 1] = np.array(board_state == -1, dtype=np.float32)
            dataset_move_probabilities[dataset_index, :, :] = probabilities
            dataset_rewards[dataset_index] = reward

        return dataset_board_states, dataset_move_probabilities, dataset_rewards

    def get_sample(self, sample_index):
        base_index = sample_index // self.symmetry_count
        symmetry_index = sample_index % self.symmetry_count
        for index, cumulative_size in enumerate(self.cumulative_dataset_sizes):
            if base_index < cumulative_size:
                intra_dataset_index = self.get_intra_dataset_index(index, base_index)
                sample = self.sample_collections[index].get(intra_dataset_index)
                augmented_sample = self.get_symmetry(sample, symmetry_index)
                return augmented_sample
        raise Exception("index out of bounds")

    def get_symmetry(self, sample, symmetry_index):
        board_state, probabilities, reward, reverse_index = sample
        board_state = np.copy(board_state)
        probabilities = np.copy(probabilities)
        if symmetry_index >= 4:
            board_state = np.flip(board_state, 0)
            probabilities = np.flip(probabilities, 0)
        rotation_index = symmetry_index % 4
        board_state = np.rot90(board_state, rotation_index)
        probabilities = np.rot90(probabilities, rotation_index)
        return (board_state, probabilities, reward, reverse_index)

    def get_intra_dataset_index(self, dataset_index, sample_index):
        if dataset_index == 0:
            offset = 0
        else:
            offset = self.cumulative_dataset_sizes[dataset_index - 1]
        return sample_index - offset


class TrainingSampleCollection:
    def __init__(self, board_states=None, move_probabilities=None, rewards=None, reverse_turn_indexes=None):
        if rewards is None:
            rewards = []
        if board_states is None:
            board_states = []
        if reverse_turn_indexes is None:
            reverse_turn_indexes = []
        if move_probabilities is None:
            move_probabilities = []
        self.board_states = board_states
        self.move_probabilities = move_probabilities
        self.rewards = rewards
        self.reverse_turn_indexes = reverse_turn_indexes

    def get_length(self):
        return len(self.rewards)

    def get_board_size(self):
        return self.board_states[0].shape

    def get(self, index):
        return self.board_states[index], self.move_probabilities[index], self.rewards[index], self.reverse_turn_indexes[
            index]

    def extend(self, training_sample_collection):
        self.board_states.extend(training_sample_collection.board_states)
        self.move_probabilities.extend(training_sample_collection.move_probabilities)
        self.rewards.extend(training_sample_collection.rewards)
        self.reverse_turn_indexes.extend(training_sample_collection.reverse_turn_indexes)

    def add_sample(self, board_state, move_probability, reward, reverse_turn_indexes):
        self.board_states.append(board_state)
        self.move_probabilities.append(move_probability)
        self.rewards.append(reward)
        self.reverse_turn_indexes.append(reverse_turn_indexes)

    def filter_samples(self, entropy_cutoff=None, turn_cutoff=None):
        if entropy_cutoff is None:
            entropy_cutoff = float("inf")
        if turn_cutoff is None:
            turn_cutoff = sys.maxsize
        original_board_states = self.board_states
        original_move_probabilities = self.move_probabilities
        original_rewards = self.rewards
        original_reverse_turn_indexes = self.reverse_turn_indexes
        self.board_states = []
        self.move_probabilities = []
        self.rewards = []
        self.reverse_turn_indexes = []

        for board_state, probability_map, reward, reverse_turn_index in zip(original_board_states,
                                                                            original_move_probabilities,
                                                                            original_rewards,
                                                                            original_reverse_turn_indexes):
            entropy = self.calculate_entropy(probability_map)
            if entropy < entropy_cutoff and reverse_turn_index <= turn_cutoff:
                self.add_sample(board_state, probability_map, reward, reverse_turn_index)

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
        reverse_turn_indexes = np.arange(len(self.board_states) - 1, -1, -1)
        return TrainingSampleCollection(self.board_states, self.move_probabilities, rewards, reverse_turn_indexes)
