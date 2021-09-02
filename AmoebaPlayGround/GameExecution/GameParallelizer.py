import os

import ray
from ray.util import ActorPool

from AmoebaPlayGround import Amoeba
from AmoebaPlayGround.Agents.AmoebaAgent import PlaceholderAgent
from AmoebaPlayGround.Agents.MCTS.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.GameExecution.GameGroup import GameGroup
from AmoebaPlayGround.GameExecution.MoveSelector import DistributionMoveSelector
from AmoebaPlayGround.GameExecution.ProgressPrinter import BaseProgressPrinter, ParallelProgressPrinter, \
    ParallelProgressPrinterActor, \
    SingleThreadedProgressPrinter
from AmoebaPlayGround.Training.Logger import Statistics
from AmoebaPlayGround.Training.TrainingSampleGenerator import SymmetricTrainingSampleGenerator, \
    TrainingSampleCollection, PlaceholderTrainingSampleGenerator


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
        print("\nBatch finished, avg_turn_time: {:.3f}\n".format(average_turn_time))


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


class ParallelGameExecutor(GameExecutor):
    def __init__(self, learning_agent: BatchMCTSAgent, reference_agent: BatchMCTSAgent, worker_count=4):

        if worker_count % 2 != 0:
            raise Exception("worker count should be the multiple of 2")
        if learning_agent.search_count != reference_agent.search_count:
            raise Exception("agents have inconsistent search counts")
        if learning_agent.batch_size != reference_agent.batch_size:
            raise Exception("agents have inconsistent batch sizes")
        if learning_agent.map_size != reference_agent.map_size:
            raise Exception("agents have inconsistent map sizes")

        os.environ['RAY_START_REDIS_WAIT_RETRIES'] = '100'
        if not ray.is_initialized():
            ray.init()
        workers = []

        self.printer_actor = ParallelProgressPrinterActor.remote(worker_count)

        for i in range(worker_count):
            worker = GameExecutorWorker.remote(learning_agent.get_weights(), reference_agent.get_weights(),
                                               learning_agent.map_size, i, learning_agent.get_config(),
                                               self.printer_actor)
            learning_agent.add_synchronized_copy(worker.set_learning_agent_weights)
            reference_agent.add_synchronized_copy(worker.set_reference_agent_weights)
            workers.append(worker)
        self.worker_pool = ActorPool(workers)
        self.worker_count = worker_count
        self.learning_agent = learning_agent
        self.reference_agent = reference_agent

    def play_games_between_agents(self, game_count, agent_1, agent_2, evaluation=False,
                                  print_progress=True):
        agent_1 = self.replace_neural_agents_with_placeholders(agent_1)
        agent_2 = self.replace_neural_agents_with_placeholders(agent_2)
        game_groups = self.generate_workloads(agent_1, agent_2, game_count, evaluation, print_progress)
        if print_progress:
            ray.get(self.printer_actor.reset.remote())
            self.group_started(agent_1.get_name(), agent_2.get_name(), game_count)
        results = self.worker_pool.map(lambda worker, params: worker.play_games_between_agents.remote(*params),
                                       game_groups)
        games, training_samples, statistics, avg_turn_time = self.merge_results(results)
        if print_progress:
            self.group_finished(avg_turn_time / self.worker_count)
        return games, training_samples, statistics

    def replace_neural_agents_with_placeholders(self, agent):
        if agent == self.learning_agent:
            agent = PlaceholderAgent("learning_agent")
        if agent == self.reference_agent:
            agent = PlaceholderAgent("reference_agent")
        return agent

    def generate_workloads(self, agent_1, agent_2, game_count, evaluation, print_progress):
        workloads = []
        games_per_group = max(1, int(game_count / self.worker_count))
        for _ in range(int(self.worker_count / 2)):
            workloads.append((games_per_group, agent_1, agent_2, evaluation, False, print_progress))
        for _ in range(int(self.worker_count / 2)):
            workloads.append((games_per_group, agent_2, agent_1, evaluation, True, print_progress))
        return workloads


@ray.remote
class GameExecutorWorker:
    def __init__(self, learning_agent_weights, reference_agent_weights, map_size, id, agent_config,
                 progress_printer_actor):
        Amoeba.map_size = map_size
        self.learning_agent = BatchMCTSAgent(**agent_config)
        self.learning_agent.set_weights(learning_agent_weights)
        self.reference_agent = BatchMCTSAgent(**agent_config)
        self.reference_agent.set_weights(reference_agent_weights)
        self.id = id
        self.progress_printer = ParallelProgressPrinter(progress_printer_actor, self.id)

    def set_learning_agent_weights(self, agent_1_weights):
        self.learning_agent.set_weights(agent_1_weights)

    def set_reference_agent_weights(self, agent_2_weights):
        self.reference_agent.set_weights(agent_2_weights)

    def play_games_between_agents(self, game_count, agent_1, agent_2, evaluation,
                                  agent_order_reversed, print_progress):

        agent_1 = self.replace_placeholder_agent(agent_1)
        agent_2 = self.replace_placeholder_agent(agent_2)
        move_selector = DistributionMoveSelector()
        if print_progress:
            progress_printer = self.progress_printer
        else:
            progress_printer = BaseProgressPrinter()
        if evaluation:
            training_sample_generator_class = PlaceholderTrainingSampleGenerator
        else:
            training_sample_generator_class = SymmetricTrainingSampleGenerator

        game_group = GameGroup(game_count, agent_1, agent_2,
                               training_sample_generator_class=training_sample_generator_class,
                               move_selector=move_selector, evaluation=evaluation,
                               reversed_agent_order=agent_order_reversed,
                               progress_printer=progress_printer)
        return game_group.play_all_games()

    def replace_placeholder_agent(self, agent):
        if agent.get_name() == "learning_agent":
            return self.learning_agent
        if agent.get_name() == "reference_agent":
            return self.reference_agent
        return agent
