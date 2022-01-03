import os
import time

import ray
from ray.util import ActorPool

from AmoebaPlayGround import Amoeba
from AmoebaPlayGround.Agents.AmoebaAgent import PlaceholderAgent
from AmoebaPlayGround.Agents.MCTS.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.GameExecution.GameGroup import GameGroup
from AmoebaPlayGround.GameExecution.MoveSelector import MoveSelectionStrategy
from AmoebaPlayGround.GameExecution.ProgressPrinter import BaseProgressPrinter, ParallelProgressPrinter, \
    ParallelProgressPrinterActor
from AmoebaPlayGround.GameExecution.SingleThreadGameExecutor import GameExecutor
from AmoebaPlayGround.Training.TrainingSampleGenerator import SymmetricTrainingSampleGenerator, \
    PlaceholderTrainingSampleGenerator


class ParallelGameExecutor(GameExecutor):
    def __init__(self, learning_agent: BatchMCTSAgent, reference_agent: BatchMCTSAgent, worker_count=4,
                 move_selection_strategy=MoveSelectionStrategy()):

        if worker_count % 2 != 0:
            raise Exception("worker count should be the multiple of 2")
        if learning_agent.search_count != reference_agent.search_count:
            raise Exception("agents have inconsistent search counts")
        if learning_agent.inference_batch_size != reference_agent.inference_batch_size:
            raise Exception("agents have inconsistent batch sizes")
        if learning_agent.map_size != reference_agent.map_size:
            raise Exception("agents have inconsistent map sizes")

        os.environ['RAY_START_REDIS_WAIT_RETRIES'] = '100'
        if not ray.is_initialized():
            ray.init()
        workers = []

        self.printer_actor = ParallelProgressPrinterActor.remote(worker_count)

        for i in range(worker_count):
            learning_agent_model = learning_agent.get_neural_network_model()
            reference_agent_model = reference_agent.get_neural_network_model()
            worker = GameExecutorWorker.remote(learning_agent_model.get_weights(), reference_agent_model.get_weights(),
                                               learning_agent.map_size, i, learning_agent.get_config(),
                                               self.printer_actor, move_selection_strategy)
            learning_agent_model.add_synchronized_copy(worker.set_learning_agent_weights)
            reference_agent_model.add_synchronized_copy(worker.set_reference_agent_weights)
            workers.append(worker)
        self.worker_pool = ActorPool(workers)
        self.worker_count = worker_count
        self.learning_agent = learning_agent
        self.reference_agent = reference_agent

    def play_games_between_agents(self, game_count, agent_1, agent_2, evaluation=False,
                                  print_progress=True):
        original_agent_1 = agent_1
        original_agent_2 = agent_2
        agent_1 = self.replace_neural_agents_with_placeholders(agent_1)
        agent_2 = self.replace_neural_agents_with_placeholders(agent_2)
        game_groups = self.generate_workloads(agent_1, agent_2, game_count, evaluation, print_progress)
        if print_progress:
            ray.get(self.printer_actor.reset.remote())
            self.group_started(agent_1.get_name(), agent_2.get_name(), game_count)
        time_before_play = time.time()
        results = self.worker_pool.map(lambda worker, params: worker.play_games_between_agents.remote(*params),
                                       game_groups)

        games, training_samples, statistics, avg_turn_time = self.merge_results(results)
        time_after_play = time.time()
        if print_progress:
            total_time = time_after_play - time_before_play
            searches_per_step = 0
            if type(original_agent_1) is BatchMCTSAgent:
                searches_per_step += original_agent_1.search_count
            if type(original_agent_2) is BatchMCTSAgent:
                searches_per_step += original_agent_2.search_count
            searches_per_step /= 2
            total_steps = statistics.step_count
            nps = (searches_per_step * total_steps) / total_time
            self.group_finished(nps)
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
                 progress_printer_actor, move_selection_strategy):
        Amoeba.map_size = map_size
        self.learning_agent = BatchMCTSAgent(**agent_config)
        learning_agent_model = self.learning_agent.get_neural_network_model()
        learning_agent_model.create_model()
        learning_agent_model.set_weights(learning_agent_weights)
        self.reference_agent = BatchMCTSAgent(**agent_config)
        reference_agent_model = self.reference_agent.get_neural_network_model()
        reference_agent_model.create_model()
        reference_agent_model.set_weights(reference_agent_weights)
        self.id = id
        self.progress_printer = ParallelProgressPrinter(progress_printer_actor, self.id)
        self.move_selection_strategy = move_selection_strategy

    def set_learning_agent_weights(self, agent_1_weights):
        self.learning_agent.get_neural_network_model().set_weights(agent_1_weights)

    def set_reference_agent_weights(self, agent_2_weights):
        self.reference_agent.get_neural_network_model().set_weights(agent_2_weights)

    def play_games_between_agents(self, game_count, agent_1, agent_2, evaluation,
                                  agent_order_reversed, print_progress):

        agent_1 = self.replace_placeholder_agent(agent_1)
        agent_2 = self.replace_placeholder_agent(agent_2)
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
                               move_selection_strategy=self.move_selection_strategy, evaluation=evaluation,
                               reversed_agent_order=agent_order_reversed,
                               progress_printer=progress_printer)
        return game_group.play_all_games()

    def replace_placeholder_agent(self, agent):
        if agent.get_name() == "learning_agent":
            return self.learning_agent
        if agent.get_name() == "reference_agent":
            return self.reference_agent
        return agent
