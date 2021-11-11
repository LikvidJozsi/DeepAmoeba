from AmoebaPlayGround.GameExecution.GameGroup import GameGroup
from AmoebaPlayGround.GameExecution.MoveSelector import DistributionMoveSelector
from AmoebaPlayGround.GameExecution.ProgressPrinter import SingleThreadedProgressPrinter, BaseProgressPrinter
from AmoebaPlayGround.Training.Logger import Statistics
from AmoebaPlayGround.Training.TrainingSampleGenerator import PlaceholderTrainingSampleGenerator, \
    SymmetricTrainingSampleGenerator, TrainingSampleCollection


class GameExecutor:

    def play_games_between_agents(self, game_count, agent_1, agent_2, evaluation, print_progress):
        pass

    def merge_results(self, game_group_results):
        combined_training_samples = TrainingSampleCollection()
        combined_statistics = Statistics()
        combined_games = []
        combined_turn_length = 0
        group_count = 0
        for game_group_result in game_group_results:
            games, training_samples, statistics, avg_turn_length = game_group_result
            combined_games.extend(games)
            combined_training_samples.extend(training_samples)
            combined_statistics.merge_statistics(statistics)
            combined_turn_length += avg_turn_length
            group_count += 1
        combined_avg_turn_length = combined_turn_length / group_count
        return combined_games, combined_training_samples, combined_statistics, combined_avg_turn_length

    def group_started(self, agent_1_name, agent_2_name, game_count):
        print("Playing {count} games between {agent_1} and {agent_2}:".
              format(count=game_count, agent_1=agent_1_name, agent_2=agent_2_name))

    def group_finished(self, average_turn_time):
        print("\nBatch finished, avg_turn_time: {:.5f}\n".format(average_turn_time))


class SingleThreadGameExecutor(GameExecutor):

    def __init__(self):
        pass

    def play_games_between_agents(self, game_count, agent_1, agent_2, evaluation=False,
                                  print_progress=True):
        move_selector = DistributionMoveSelector()
        games_per_group = max(1, int(game_count / 2))
        if evaluation:
            training_sample_generator_class = PlaceholderTrainingSampleGenerator
        else:
            training_sample_generator_class = SymmetricTrainingSampleGenerator

        if print_progress:
            progress_printer = SingleThreadedProgressPrinter()
        else:
            progress_printer = BaseProgressPrinter()
        game_group_1 = GameGroup(games_per_group, agent_1, agent_2, None, progress_printer=progress_printer,
                                 training_sample_generator_class=training_sample_generator_class,
                                 move_selector=move_selector, evaluation=evaluation)
        game_group_2 = GameGroup(games_per_group, agent_2, agent_1, None, progress_printer=progress_printer,
                                 training_sample_generator_class=training_sample_generator_class,
                                 move_selector=move_selector, evaluation=evaluation,
                                 reversed_agent_order=True)

        if print_progress:
            self.group_started(agent_1.get_name(), agent_2.get_name(), games_per_group)
        group_1_results = game_group_1.play_all_games()
        if print_progress:
            self.group_started(agent_2.get_name(), agent_1.get_name(), games_per_group)
        group_2_results = game_group_2.play_all_games()
        games, training_samples, statistics, avg_turn_length = self.merge_results([group_1_results, group_2_results])
        if print_progress:
            self.group_finished(avg_turn_length)
        return games, training_samples, statistics
