import math
from typing import List

from AmoebaPlayGround.Amoeba import AmoebaGame, Move
from AmoebaPlayGround.GameBoard import Player


class RewardCalculator():
    def get_training_data(self, games: List[AmoebaGame]):
        pass


class TrainingSample:
    def __init__(self, move: Move, reward: float):
        self.board_state = move.board_state
        self.step = move.step
        self.reward = reward

    def __str__(self):
        return str(self.step) + " " + str(self.reward)

    def unpack(training_samples: List['TrainingSample']):
        board_states = list(map(lambda sample: sample.board_state, training_samples))
        steps = list(map(lambda sample: sample.step, training_samples))
        rewards = list(map(lambda sample: sample.reward, training_samples))
        return board_states, steps, rewards



class PolicyGradients(RewardCalculator):
    def __init__(self, discount_factor=0.6, reward_for_win=1.2, reward_for_loss=-1,
                 reward_for_tie=-0.5, reward_cutoff_threshold=0.05, teach_with_draws=False, teach_with_losses=False):
        self.discount_factor = discount_factor
        self.reward_for_win = reward_for_win
        self.reward_for_loss = reward_for_loss
        self.reward_for_tie = reward_for_tie
        self.reward_cutoff_threshold = reward_cutoff_threshold
        self.teach_with_draws = teach_with_draws
        self.teach_with_losses = teach_with_losses

    def get_training_data(self, games: List[AmoebaGame]) -> List[TrainingSample]:
        training_samples = []
        for game in games:
            training_samples.extend(self._get_training_samples_from_game(game))
        return training_samples

    def _get_training_samples_from_game(self, game: AmoebaGame) -> List[TrainingSample]:
        steps_needed = self._get_num_of_steps_needed()
        moves = game.get_last_moves(steps_needed)
        winner = game.winner
        if self.teach_with_draws and winner == Player.NOBODY:
            return self._get_training_samples_from_draw(self, moves)
        elif winner != Player.NOBODY:
            winner_moves = filter(lambda move: move.player == winner, moves)
            winner_samples = self._get_training_samples(winner_moves, self.reward_for_win)
            samples = winner_samples
            samples.extend(self._get_loser_training_samples(moves, winner))

            return samples
        return []

    def _get_loser_training_samples(self, moves, winner):
        loser_samples = []
        if self.teach_with_losses:
            loser_moves = filter(lambda move: move.player != winner, moves)
            loser_samples = self._get_training_samples(loser_moves, self.reward_for_loss)
        return loser_samples

    def _get_training_samples(self, moves, reward):
        discounted_reward = reward
        training_samples = []
        for move in moves:
            sample = TrainingSample(move, discounted_reward)
            discounted_reward = discounted_reward * self.discount_factor
            training_samples.append(sample)
        return training_samples

    def _get_training_samples_from_draw(self, moves, reward):
        discounted_reward = reward
        training_samples = []
        for index, move in enumerate(moves):
            sample = TrainingSample(move, discounted_reward)
            if index % 2 == 1:
                discounted_reward = discounted_reward * self.discount_factor
            training_samples.append(sample)
        return training_samples

    def _get_num_of_steps_needed(self):
        # if the reward is close to 0 it is not worth it to add to training, determine how many steps
        # it takes for it to fade
        steps_needed = math.floor(math.log(self.reward_cutoff_threshold, self.discount_factor))
        # multiplied by two because since there are two players it takes rewards get decreased every two turns
        return int(steps_needed * 2)


class PolicyGradientsWithNegativeTeaching(PolicyGradients):
    def __init__(self, discount_factor=0.7, nr_of_alternative_steps=5, reward_for_alt_move=0.12):
        super().__init__(discount_factor=discount_factor)
        self.nr_of_alternative_steps = nr_of_alternative_steps
        self.reward_for_alt_move = reward_for_alt_move

    def _get_loser_training_samples(self, moves, winner):
        loser_alternative_moves = []

        nr_of_relevant = min(2 * self.nr_of_alternative_steps, len(moves))
        relevant_moves = moves[:nr_of_relevant]

        for winner_move, loser_move in zip(relevant_moves[0::2], relevant_moves[1::2]):
            loser_alternative_moves.append(Move(loser_move.board_state,
                                                winner_move.step, loser_move.player))

        return self._get_training_samples(loser_alternative_moves, reward=self.reward_for_alt_move)
