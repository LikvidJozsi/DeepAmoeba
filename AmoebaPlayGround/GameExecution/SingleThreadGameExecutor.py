import time

from AmoebaPlayGround.Agents.MCTS.MCTSAgent import MCTSAgent
from AmoebaPlayGround.GameExecution.GameGroup import GameGroup
from AmoebaPlayGround.GameExecution.MoveSelector import MoveSelectionStrategy
from AmoebaPlayGround.GameExecution.ProgressPrinter import SingleThreadedProgressPrinter, BaseProgressPrinter
from AmoebaPlayGround.Training.Logger import Statistics
from AmoebaPlayGround.Training.TrainingSampleGenerator import PlaceholderTrainingSampleGenerator, \
    SymmetricTrainingSampleGenerator, TrainingSampleCollection


class GameExecutor:

    def play_games_between_agents(self, game_count, agent_1, agent_2, map_size, evaluation, print_progress):
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
        combined_avg_turn_length = combined_turn_length / group_count / group_count
        return combined_games, combined_training_samples, combined_statistics, combined_avg_turn_length

    def group_started(self, agent_1_name, agent_2_name, game_count):
        print("Playing {count} games between {agent_1} and {agent_2}:".
              format(count=game_count, agent_1=agent_1_name, agent_2=agent_2_name))

    def group_finished(self, average_nodes_per_sec):
        print("\nBatch finished, avg_nps: {:.5f}\n".format(average_nodes_per_sec))


class SingleThreadGameExecutor(GameExecutor):

    def __init__(self, move_selection_strategy=MoveSelectionStrategy()):
        self.move_selection_strategy = move_selection_strategy

    def play_games_between_agents(self, game_count, agent_1, agent_2, map_size, evaluation=False,
                                  print_progress=True):
        games_per_group = max(1, int(game_count / 2))
        if evaluation:
            training_sample_generator_class = PlaceholderTrainingSampleGenerator
        else:
            training_sample_generator_class = SymmetricTrainingSampleGenerator

        if print_progress:
            progress_printer = SingleThreadedProgressPrinter()
        else:
            progress_printer = BaseProgressPrinter()

        inference_batch_size = agent_1.get_neural_network_model().inference_batch_size
        intra_game_parallelism = agent_1.max_intra_game_parallelism
        max_parallel_games = int(inference_batch_size / intra_game_parallelism)

        time_before_play = time.perf_counter()
        game_group_1 = GameGroup(games_per_group, max_parallel_games, map_size, agent_1, agent_2,
                                 progress_printer=progress_printer,
                                 training_sample_generator_class=training_sample_generator_class,
                                 move_selection_strategy=self.move_selection_strategy)
        game_group_2 = GameGroup(games_per_group, max_parallel_games, map_size, agent_2, agent_1,
                                 progress_printer=progress_printer,
                                 training_sample_generator_class=training_sample_generator_class,
                                 move_selection_strategy=self.move_selection_strategy,
                                 reversed_agent_order=True)

        if print_progress:
            self.group_started(agent_1.get_name(), agent_2.get_name(), games_per_group)
        group_1_results = game_group_1.play_all_games()
        if print_progress:
            self.group_started(agent_2.get_name(), agent_1.get_name(), games_per_group)
        group_2_results = game_group_2.play_all_games()
        games, training_samples, statistics, avg_turn_length = self.merge_results([group_1_results, group_2_results])
        time_after_play = time.perf_counter()
        if print_progress:
            total_time = time_after_play - time_before_play

            searches_per_step = self.get_search_count(agent_1, agent_2)
            total_steps = statistics.step_count
            nps = (searches_per_step * total_steps) / total_time
            self.group_finished(nps)
        return games, training_samples, statistics

    def get_search_count(self, agent_1, agent_2):
        if isinstance(agent_1, MCTSAgent):
            if isinstance(agent_2, MCTSAgent):
                return (agent_1.search_count + agent_2.search_count) / 2
            else:
                return agent_1.search_count
        else:
            if isinstance(agent_2, MCTSAgent):
                return agent_2.search_count
            else:
                print("npm is not calculable since there are no MCTSAgents present")
                return 1
