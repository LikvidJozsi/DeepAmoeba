import time

from AmoebaPlayGround.Amoeba import AmoebaGame, Player
from AmoebaPlayGround.GameExecution.MoveSelector import MoveSelectionStrategy
from AmoebaPlayGround.GameExecution.ProgressPrinter import BaseProgressPrinter
from AmoebaPlayGround.Training.Logger import Statistics
from AmoebaPlayGround.Training.TrainingSampleGenerator import SymmetricTrainingSampleGenerator, TrainingSampleCollection


class GameGroup:
    def __init__(self, total_game_count, batch_size, x_agent=None, o_agent=None,
                 view=None, training_sample_generator_class=SymmetricTrainingSampleGenerator,
                 move_selection_strategy=MoveSelectionStrategy(), evaluation=False, reversed_agent_order=False,
                 progress_printer=BaseProgressPrinter()):
        self.reversed_agent_order = reversed_agent_order
        self.x_agent = x_agent
        self.o_agent = o_agent
        self.progress_printer = progress_printer
        self.games = []
        self.move_selection_strategy = move_selection_strategy
        self.training_sample_generators = []
        self.evaluation = evaluation
        self.total_game_count = total_game_count
        self.training_sample_generator_class = training_sample_generator_class
        self.batch_size = batch_size
        if self.total_game_count < batch_size:
            batch_size = self.total_game_count
        for index in range(batch_size):
            self.games.append(AmoebaGame(view))
            self.training_sample_generators.append(training_sample_generator_class())

    def set_x_agent(self, x_agent):
        self.x_agent = x_agent

    def set_o_agent(self, o_agent):
        self.o_agent = o_agent

    def play_all_games(self):
        finished_games = []
        number_of_games = self.total_game_count
        training_samples = TrainingSampleCollection()
        sum_turn_length_sec = 0
        turn_number = 0
        statistics = Statistics()
        unstarted_games_remaining = self.total_game_count - self.batch_size
        turn = 0

        while len(self.games) != 0 or unstarted_games_remaining > 0:
            if turn % 2 == 0 and len(self.games) < self.batch_size and unstarted_games_remaining > 0:
                for i in range(min(self.batch_size - len(self.games), unstarted_games_remaining)):
                    self.games.append(AmoebaGame())
                    self.training_sample_generators.append(self.training_sample_generator_class())

            if len(self.games) != 0:
                next_agent = self.get_next_agent(
                    self.games[0])  # the same agent has its turn in every active game at the
            else:
                next_agent = self.get_next_agent(AmoebaGame())
            # same time, therfore getting the agent of any of them is enough
            time_before_step = time.perf_counter()
            action_probabilities, step_statistics = next_agent.get_step(self.games, self.games[0].get_next_player(),
                                                                        self.evaluation)
            statistics.merge_statistics(step_statistics)
            time_after_step = time.perf_counter()
            for game, training_sample_generator, action_probability_map in zip(self.games,
                                                                               self.training_sample_generators,
                                                                               action_probabilities):
                action = self.move_selection_strategy.get_move_selector(turn_number, self.evaluation).select_move(
                    action_probability_map)
                training_sample_generator.add_move(game.get_board_of_next_player(), action_probability_map,
                                                   game.get_next_player())
                game.step(action)
                if game.has_game_ended():
                    finished_games.append(game)
                    training_samples_from_game = training_sample_generator.get_training_data(game.winner)
                    training_samples.extend(training_samples_from_game)
            turn_length_per_game_sec = (time_after_step - time_before_step) / len(self.games)
            sum_turn_length_sec += turn_length_per_game_sec
            self.games = [game for game in self.games if not game in finished_games]
            turn_number += 1
            self.progress_printer.print_progress(len(finished_games) / number_of_games, turn_number,
                                                 turn_length_per_game_sec, step_statistics)
            turn += 1

        avg_time_per_turn_per_game = sum_turn_length_sec / turn_number
        statistics.aggregate_game_length = self.get_aggregate_game_length(finished_games)
        statistics.game_count = len(finished_games)
        games_player_1_won, games_player_2_won, draws, games_x_won = self.get_win_statistics(finished_games)
        statistics.add_win_statistics(games_player_1_won, games_player_2_won, draws, games_x_won)
        return finished_games, training_samples, statistics, avg_time_per_turn_per_game

    def get_win_statistics(self, games):
        games_player_1_won = 0
        games_player_2_won = 0
        games_draw = 0
        for game in games:
            winner = game.winner
            if winner == Player.X:
                games_player_1_won += 1
            elif winner == Player.O:
                games_player_2_won += 1
            else:
                games_draw += 1
        if self.reversed_agent_order:
            return games_player_2_won, games_player_1_won, games_draw, games_player_1_won
        else:
            return games_player_1_won, games_player_2_won, games_draw, games_player_1_won

    def get_aggregate_game_length(self, games):
        sum_game_length = 0
        for game in games:
            sum_game_length += game.num_steps
        return sum_game_length

    def get_next_agent(self, game):
        if game.previous_player == Player.X:
            return self.o_agent
        else:
            return self.x_agent
