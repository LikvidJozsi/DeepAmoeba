import time

from AmoebaPlayGround.Amoeba import AmoebaGame, Player
from AmoebaPlayGround.GameExecution.MoveSelector import MaximalMoveSelector
from AmoebaPlayGround.GameExecution.ProgressPrinter import BaseProgressPrinter
from AmoebaPlayGround.Training.Logger import Statistics
from AmoebaPlayGround.Training.TrainingSampleGenerator import SymmetricTrainingSampleGenerator, TrainingSampleCollection


class GameGroup:
    def __init__(self, batch_size, x_agent=None, o_agent=None,
                 view=None, training_sample_generator_class=SymmetricTrainingSampleGenerator,
                 move_selector=MaximalMoveSelector(), evaluation=False, reversed_agent_order=False,
                 progress_printer=BaseProgressPrinter()):
        self.reversed_agent_order = reversed_agent_order
        self.x_agent = x_agent
        self.o_agent = o_agent
        self.progress_printer = progress_printer
        self.games = []
        self.move_selector = move_selector
        self.training_sample_generators = []
        self.evaluation = evaluation
        for index in range(batch_size):
            self.games.append(AmoebaGame(view))
            self.training_sample_generators.append(training_sample_generator_class())

    def set_x_agent(self, x_agent):
        self.x_agent = x_agent

    def set_o_agent(self, o_agent):
        self.o_agent = o_agent

    def play_all_games(self):
        finished_games = []
        number_of_games = len(self.games)
        training_samples = TrainingSampleCollection()
        sum_turn_length_sec = 0
        turn_number = 0
        statistics = Statistics()
        while len(self.games) != 0:
            next_agent = self.get_next_agent(self.games[0])  # the same agent has its turn in every active game at the
            # same time, therfore getting the agent of any of them is enough
            time_before_step = time.time()
            action_probabilities, step_statistics = next_agent.get_step(self.games, self.games[0].get_next_player(),
                                                                        self.evaluation)
            statistics.merge_statistics(step_statistics)
            time_after_step = time.time()
            for game, training_sample_generator, action_probability_map in zip(self.games,
                                                                               self.training_sample_generators,
                                                                               action_probabilities):
                action = self.move_selector.select_move(action_probability_map)
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

        avg_time_per_turn_per_game = sum_turn_length_sec / turn_number
        statistics.aggregate_game_length = self.get_aggregate_game_length(finished_games)
        statistics.game_count = len(finished_games)
        games_player_1_won, games_player_2_won, draws = self.get_win_statistics(finished_games)
        statistics.add_win_statistics(games_player_1_won, games_player_2_won, draws)
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
            return games_player_2_won, games_player_1_won, games_draw
        else:
            return games_player_1_won, games_player_2_won, games_draw

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
