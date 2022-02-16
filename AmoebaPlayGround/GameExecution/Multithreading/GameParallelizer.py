import os
import time

import ray
from ray.util import ActorPool

from AmoebaPlayGround import Amoeba
from AmoebaPlayGround.Agents.AmoebaAgent import PlaceholderAgent
from AmoebaPlayGround.Agents.MCTS.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.GameExecution.GameGroup import GameGroup
from AmoebaPlayGround.GameExecution.MoveSelector import MoveSelectionStrategy
from AmoebaPlayGround.GameExecution.Multithreading.InferenceServer import InferenceServer, InferenceServerWrapper
from AmoebaPlayGround.GameExecution.ProgressPrinter import BaseProgressPrinter, ParallelProgressPrinter, \
    ParallelProgressPrinterActor
from AmoebaPlayGround.GameExecution.SingleThreadGameExecutor import GameExecutor
from AmoebaPlayGround.Training.TrainingSampleGenerator import SymmetricTrainingSampleGenerator, \
    PlaceholderTrainingSampleGenerator


class ParallelGameExecutor(GameExecutor):
    def __init__(self, learning_agent: BatchMCTSAgent, reference_agent: BatchMCTSAgent, worker_count=4,
                 inference_server_count=2, move_selection_strategy=MoveSelectionStrategy()):

        if worker_count % 2 != 0:
            raise Exception("worker count should be the multiple of 2")
        if learning_agent.search_count != reference_agent.search_count:
            raise Exception("agents have inconsistent search counts")
        if learning_agent.search_batch_size != reference_agent.search_batch_size:
            raise Exception("agents have inconsistent batch sizes")
        if learning_agent.map_size != reference_agent.map_size:
            raise Exception("agents have inconsistent map sizes")
        if worker_count % (inference_server_count * 2) != 0:
            raise Exception("worker count must be divisible by inference server count*2")

        os.environ['RAY_START_REDIS_WAIT_RETRIES'] = '100'
        if not ray.is_initialized():
            ray.init()
        workers = []

        learning_agent_model = learning_agent.get_neural_network_model()
        reference_agent_model = reference_agent.get_neural_network_model()

        inference_batch_size = learning_agent_model.inference_batch_size
        per_worker_batch_size = inference_batch_size / worker_count * 2 * inference_server_count
        print(f"{per_worker_batch_size} parallel searches per worker")
        self.printer_actor = ParallelProgressPrinterActor.remote(worker_count)

        self.inference_servers = []
        for i in range(inference_server_count):
            inference_server = InferenceServer.remote(learning_agent.model.get_skeleton(),
                                                      reference_agent.model.get_skeleton())
            learning_agent_model.add_synchronized_copy(inference_server.set_learning_agent_weights)
            reference_agent_model.add_synchronized_copy(inference_server.set_reference_agent_weights)
            self.inference_servers.append(inference_server)

        self.workers_per_server = int(worker_count / inference_server_count)
        for server_index, inference_server in enumerate(self.inference_servers):
            for worker_index in range(self.workers_per_server):
                worker_id = server_index * self.workers_per_server + worker_index
                worker = self.create_worker(learning_agent, reference_agent, per_worker_batch_size, worker_id,
                                            move_selection_strategy,
                                            inference_server)
                workers.append(worker)
        self.worker_pool = ActorPool(workers)
        self.worker_count = worker_count
        self.learning_agent = learning_agent
        self.reference_agent = reference_agent

    def create_worker(self, learning_agent, reference_agent, per_worker_batch_size, id, move_selection_strategy,
                      inference_server):
        learning_agent_predictor = InferenceServerWrapper("learning_agent", inference_server)
        learning_agent_copy = learning_agent.get_copy_without_model()
        learning_agent_copy.search_batch_size = per_worker_batch_size
        learning_agent_copy.model = learning_agent_predictor
        reference_agent_predictor = InferenceServerWrapper("reference_agent", inference_server)
        reference_agent_copy = reference_agent.get_copy_without_model()
        reference_agent_copy.search_batch_size = per_worker_batch_size
        reference_agent_copy.model = reference_agent_predictor
        worker = GameExecutorWorker.remote(learning_agent_copy, reference_agent_copy,
                                           learning_agent.map_size, id,
                                           self.printer_actor, move_selection_strategy)
        return worker

    def play_games_between_agents(self, game_count, agent_1, agent_2, evaluation=False,
                                  print_progress=True):
        original_agent_1 = agent_1
        original_agent_2 = agent_2
        agent_1 = self.replace_neural_agents_with_placeholders(agent_1)
        agent_2 = self.replace_neural_agents_with_placeholders(agent_2)
        game_groups = self.generate_workloads(agent_1, agent_2, game_count, evaluation, print_progress)

        for inference_server in self.inference_servers:
            ray.get(inference_server.game_group_started.remote(self.workers_per_server))

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
    def __init__(self, learning_agent, reference_agent, map_size, id, progress_printer_actor, move_selection_strategy):
        Amoeba.map_size = map_size
        self.learning_agent = learning_agent
        self.reference_agent = reference_agent
        self.id = id
        self.progress_printer = ParallelProgressPrinter(progress_printer_actor, self.id)
        self.move_selection_strategy = move_selection_strategy

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
        results = game_group.play_all_games()
        self.learning_agent.model.worker_finished()
        return results

    def replace_placeholder_agent(self, agent):
        if agent.get_name() == "learning_agent":
            return self.learning_agent
        if agent.get_name() == "reference_agent":
            return self.reference_agent
        return agent
