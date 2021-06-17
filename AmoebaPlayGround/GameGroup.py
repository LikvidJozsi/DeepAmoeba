import sys
import time

from AmoebaPlayGround.Amoeba import AmoebaGame, Player
from AmoebaPlayGround.Logger import Statistics
from AmoebaPlayGround.MoveSelector import MaximalMoveSelector
from AmoebaPlayGround.TrainingSampleGenerator import SymmetricTrainingSampleGenerator, TrainingSampleCollection


class GameGroup:
    def __init__(self, batch_size, x_agent, o_agent,
                 view=None, training_sample_generator_class=SymmetricTrainingSampleGenerator, log_progress=False,
                 move_selector=MaximalMoveSelector()):
        self.x_agent = x_agent
        self.o_agent = o_agent
        self.log_progress = log_progress
        self.games = []
        self.move_selector = move_selector
        self.training_sample_generators = []
        for index in range(batch_size):
            self.games.append(AmoebaGame(view))
            self.training_sample_generators.append(training_sample_generator_class())

    def play_all_games(self):
        finished_games = []
        number_of_games = len(self.games)
        training_samples = TrainingSampleCollection()
        avg_turn_length_sec = 0
        turn_number = 0
        statistics = Statistics()
        if self.log_progress:
            print("Playing {count} games between {agent_1} and {agent_2}:".
                  format(count=number_of_games, agent_1=self.x_agent.get_name(), agent_2=self.o_agent.get_name()))
        while len(self.games) != 0:
            next_agent = self.get_next_agent(self.games[0])  # the same agent has its turn in every active game at the
            # same time, therfore getting the agent of any of them is enough
            maps = self.get_maps_of_games()
            time_before_step = time.time()
            action_probabilities, step_statistics = next_agent.get_step(maps, self.games[0].get_next_player())
            statistics.merge_statistics(step_statistics)
            time_after_step = time.time()
            for game, training_sample_generator, action_probabilities in zip(self.games,
                                                                             self.training_sample_generators,
                                                                             action_probabilities):
                action = self.move_selector.select_move(action_probabilities)
                game.step(action)
                training_sample_generator.add_move(game.get_board_of_previous_player(), action_probabilities,
                                                   game.previous_player)
                if game.has_game_ended():
                    finished_games.append(game)
                    training_samples_from_game = training_sample_generator.get_training_data(game.winner)
                    training_samples.extend(training_samples_from_game)
            turn_length_per_game_sec = (time_after_step - time_before_step) / len(self.games)
            self.games = [game for game in self.games if not game in finished_games]
            turn_number += 1

            avg_turn_length_sec = avg_turn_length_sec * (
                    turn_number - 1) / turn_number + turn_length_per_game_sec / turn_number
            self.print_progress(len(finished_games) / number_of_games, turn_number, turn_length_per_game_sec,
                                step_statistics)

        if self.log_progress:
            print("Batch finished, avg_turn_time: {:.2f}".format(avg_turn_length_sec))
        statistics.aggregate_game_length = self.get_aggregate_game_length(finished_games)
        statistics.game_count = len(finished_games)
        return finished_games, training_samples, statistics

    def get_aggregate_game_length(self, games):
        sum_game_length = 0
        for game in games:
            sum_game_length += game.num_steps
        return sum_game_length

    def get_maps_of_games(self):
        maps = []
        for game in self.games:
            maps.append(game.map)
        return maps

    def get_next_agent(self, game):
        if game.previous_player == Player.X:
            return self.o_agent
        else:
            return self.x_agent

    def print_progress(self, progress, turn, time, extra_search_data):
        if self.log_progress:
            barLength = 20
            status = "in progress"
            if progress >= 1:
                progress = 1
                status = "done\r\n"
            block = int(round(barLength * progress))
            text = "\r[{0}] {1:.1f}%, turn number: {2} , turn time: {3:.4f}, {4}, status: {5}".format(
                "#" * block + "-" * (barLength - block), progress * 100, turn, time, extra_search_data, status)
            sys.stdout.write(text)
            sys.stdout.flush()
