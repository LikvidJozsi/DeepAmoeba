import time

from AmoebaPlayGround.Amoeba import AmoebaGame, Player
from AmoebaPlayGround.GameExecution.MoveSelector import MoveSelectionStrategy
from AmoebaPlayGround.GameExecution.ProgressPrinter import BaseProgressPrinter
from AmoebaPlayGround.Training.Logger import Statistics
from AmoebaPlayGround.Training.TrainingSampleGenerator import SymmetricTrainingSampleGenerator, TrainingSampleCollection


class GameGroup:
    def __init__(self, total_game_count, batch_size, map_size, x_agent=None, o_agent=None,
                 training_sample_generator_class=SymmetricTrainingSampleGenerator,
                 move_selection_strategy=MoveSelectionStrategy(), reversed_agent_order=False,
                 progress_printer=BaseProgressPrinter()):
        self.reversed_agent_order = reversed_agent_order
        self.x_agent = x_agent
        self.o_agent = o_agent
        self.progress_printer = progress_printer
        self.games = []
        self.move_selection_strategy = move_selection_strategy
        self.training_sample_generators = []
        self.total_game_count = total_game_count
        self.training_sample_generator_class = training_sample_generator_class
        self.batch_size = batch_size
        self.statistics = Statistics()
        self.finished_games = []
        self.training_samples = TrainingSampleCollection()
        if self.total_game_count < batch_size:
            batch_size = self.total_game_count
        for index in range(batch_size):
            self.games.append(AmoebaGame(map_size))
            self.training_sample_generators.append(training_sample_generator_class())

    def set_x_agent(self, x_agent):
        self.x_agent = x_agent

    def set_o_agent(self, o_agent):
        self.o_agent = o_agent

    def play_all_games(self):
        number_of_games = self.total_game_count
        sum_turn_length_sec = 0
        unstarted_games_remaining = self.total_game_count - self.batch_size
        turn = 0

        while len(self.games) != 0 or unstarted_games_remaining > 0:
            self.fill_game_count_to_capacity(turn, unstarted_games_remaining)
            next_agent = self.get_next_agent(self.games[0])  # the same agent has its turn in every active game at the
            # same time, therfore getting the agent of any of them is enough
            action_probabilities, step_statistics, calculation_time = self.calculate_action_probabilities(next_agent)
            self.make_moves(action_probabilities, turn)
            turn_length_per_game_sec = calculation_time / len(self.games)
            sum_turn_length_sec += turn_length_per_game_sec
            self.games = self.filter_active_games()
            self.progress_printer.print_progress(len(self.finished_games) / number_of_games, turn,
                                                 turn_length_per_game_sec, step_statistics)
            turn += 1

        avg_time_per_turn_per_game = sum_turn_length_sec / turn
        self.compile_statistics()
        return self.finished_games, self.training_samples, self.statistics, avg_time_per_turn_per_game

    def fill_game_count_to_capacity(self, turn, unstarted_games_remaining):
        if turn % 2 == 0 or len(self.games) == 0:
            for i in range(min(self.batch_size - len(self.games), unstarted_games_remaining)):
                self.games.append(AmoebaGame())
                self.training_sample_generators.append(self.training_sample_generator_class())

    def calculate_action_probabilities(self, next_agent):
        time_before_step = time.perf_counter()
        action_probabilities, step_statistics = next_agent.get_step(self.games, self.games[0].get_next_player())
        self.statistics.merge_statistics(step_statistics)
        time_after_step = time.perf_counter()
        caluclation_duration = time_after_step - time_before_step
        return action_probabilities, step_statistics, caluclation_duration

    def make_moves(self, action_probabilities, turn):
        for game, training_sample_generator, action_probability_map in zip(self.games,
                                                                           self.training_sample_generators,
                                                                           action_probabilities):
            action = self.move_selection_strategy.get_move_selector(turn).select_move(
                action_probability_map)
            training_sample_generator.add_move(game.get_board_of_next_player(), action_probability_map,
                                               game.get_next_player())
            game.step(action)

    def filter_active_games(self):
        active_games = []
        for game, training_sample_generator in zip(self.games, self.training_sample_generators):
            if game.has_game_ended():
                self.finished_games.append(game)
                training_samples_from_game = training_sample_generator.get_training_data(game.winner)
                self.training_samples.extend(training_samples_from_game)
            else:
                active_games.append(game)
        return active_games

    def compile_statistics(self):
        self.statistics.aggregate_game_length = self.get_aggregate_game_length()
        self.statistics.game_count = len(self.finished_games)
        games_player_1_won, games_player_2_won, draws, games_x_won = self.get_win_statistics()
        self.statistics.add_win_statistics(games_player_1_won, games_player_2_won, draws, games_x_won)

    def get_win_statistics(self):
        games_player_1_won = 0
        games_player_2_won = 0
        games_draw = 0
        for game in self.finished_games:
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

    def get_aggregate_game_length(self):
        sum_game_length = 0
        for game in self.finished_games:
            sum_game_length += game.num_steps
        return sum_game_length

    def get_next_agent(self, game):
        if game.previous_player == Player.X:
            return self.o_agent
        else:
            return self.x_agent
