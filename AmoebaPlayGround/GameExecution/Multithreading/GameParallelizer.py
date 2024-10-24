import os
import time

import ray
from ray.util import ActorPool

from AmoebaPlayGround.Agents.MCTS.MCTSAgent import MCTSAgent
from AmoebaPlayGround.GameExecution.GameGroup import GameGroup
from AmoebaPlayGround.GameExecution.MoveSelector import MOVE_SELECTION_STRATEGIES
from AmoebaPlayGround.GameExecution.Multithreading.InferenceServer import InferenceServer, InferenceServerWrapper
from AmoebaPlayGround.GameExecution.ProgressPrinter import BaseProgressPrinter, ParallelProgressPrinter, \
    ParallelProgressTrackerActor, print_progress
from AmoebaPlayGround.GameExecution.SingleThreadGameExecutor import GameExecutor
from AmoebaPlayGround.Training.TrainingSampleGenerator import SymmetricTrainingSampleGenerator, \
    PlaceholderTrainingSampleGenerator


def init_ray_if_needed():
    os.environ['RAY_START_REDIS_WAIT_RETRIES'] = '100'
    if not ray.is_initialized():
        ray.init()


class ParallelGameExecutor(GameExecutor):
    def __init__(self, learning_agent: MCTSAgent, reference_agent: MCTSAgent, config):

        if config['workers_per_inference_server'] % 2 != 0:
            raise Exception("worker per inference server count should be the multiple of 2")

        init_ray_if_needed()

        self.workers_per_server = config['workers_per_inference_server']
        self.inference_server_count = config['inference_server_count']

        self.worker_count = self.workers_per_server * self.inference_server_count
        self.progress_tracker_actor = ParallelProgressTrackerActor.remote(self.worker_count)

        self.per_worker_batch_size = config["inference_batch_size"] / self.workers_per_server * 2
        print(f"{self.per_worker_batch_size} parallel searches per worker")

        intra_game_parallelism = learning_agent.max_intra_game_parallelism
        self.max_parallel_games = int(self.per_worker_batch_size / intra_game_parallelism)

        self.create_inference_servers(learning_agent, reference_agent)
        self.create_mcts_workers(config)

    def create_mcts_workers(self, config):
        workers = []
        move_selection_strategy = MOVE_SELECTION_STRATEGIES[config["move_selection_strategy_type"]]

        for server_index, inference_server in enumerate(self.inference_servers):
            for worker_index in range(self.workers_per_server):
                worker_id = server_index * self.workers_per_server + worker_index
                worker = GameExecutorWorker.remote(worker_id, self.progress_tracker_actor, move_selection_strategy)
                workers.append(worker)
        self.worker_pool = ActorPool(workers)

    def create_inference_servers(self, learning_agent, reference_agent):
        self.inference_servers = []
        for i in range(self.inference_server_count):
            inference_server = InferenceServer.remote({"agent_1": learning_agent.model.get_packed_copy(),
                                                       "agent_2": reference_agent.model.get_packed_copy()})
            self.inference_servers.append(inference_server)
        self.inference_server_pool = ActorPool(self.inference_servers)

    def play_games_between_agents(self, game_count, agent_1, agent_2, map_size, evaluation=False,
                                  print_progress_active=True):
        self.distribute_weights(agent_1, agent_2)
        game_groups = self.generate_workloads(agent_1, agent_2, map_size, game_count, self.max_parallel_games,
                                              evaluation,
                                              print_progress_active)

        for inference_server in self.inference_servers:
            ray.get(inference_server.game_group_started.remote(self.workers_per_server))

        if print_progress_active:
            ray.get(self.progress_tracker_actor.reset.remote())
            self.group_started(agent_1.get_name(), agent_2.get_name(), game_count)
        time_before_play = time.time()
        if agent_1 == agent_2:
            results = self.worker_pool.map(lambda worker, params: worker.play_self_play_games.remote(*params),
                                           game_groups)
        else:
            results = self.worker_pool.map(lambda worker, params: worker.play_games_between_agents.remote(*params),
                                           game_groups)

        while not ray.get(self.progress_tracker_actor.have_all_games_finished.remote()):
            print_progress(*ray.get(self.progress_tracker_actor.get_combined_metrics.remote()))
            time.sleep(0.5)

        games, training_samples, statistics, avg_turn_time = self.merge_results(results)
        time_after_play = time.time()
        if print_progress_active:
            total_time = time_after_play - time_before_play
            searches_per_step = 0
            if type(agent_1) is MCTSAgent:
                searches_per_step += agent_1.search_count
            if type(agent_2) is MCTSAgent:
                searches_per_step += agent_2.search_count
            searches_per_step /= 2
            total_steps = statistics.step_count
            nps = (searches_per_step * total_steps) / total_time
            self.group_finished(nps)
        return games, training_samples, statistics

    def distribute_weights(self, agent_1, agent_2):
        self.distribute_weights_for_agent("agent_1", agent_1)
        self.distribute_weights_for_agent("agent_2", agent_2)

    def distribute_weights_for_agent(self, agent_name, agent):
        if type(agent) is MCTSAgent:
            results = self.inference_server_pool.map(
                lambda inference_server, _: inference_server.set_agent_weights.remote(agent_name,
                                                                                      agent.model.get_weights()),
                [None] * len(self.inference_servers))
            for result in results:
                pass

    def get_modified_agent_copy(self, agent, agent_name, inference_server):
        agent_with_inference = agent.get_copy_without_model()
        agent_with_inference.search_batch_size = self.per_worker_batch_size
        agent_with_inference.model = InferenceServerWrapper(agent_name, inference_server)
        return agent_with_inference

    def generate_workloads(self, agent_1, agent_2, map_size, game_count, max_parallel_games, evaluation,
                           print_progress):
        workloads = []

        games_per_group = max(1, int(game_count / self.worker_count)) # TODO handle case where there are more workers than games
        leftovers = max(0, game_count - games_per_group * self.worker_count)
        for inference_server in self.inference_servers:
            if type(agent_1) is MCTSAgent:
                agent_1 = self.get_modified_agent_copy(agent_1, "agent_1", inference_server)
            if type(agent_2) is MCTSAgent:
                agent_2 = self.get_modified_agent_copy(agent_2, "agent_2", inference_server)

            for _ in range(int(self.workers_per_server / 2)):
                if leftovers > 0:
                    games_in_group = games_per_group + 1
                    leftovers -= 1
                else:
                    games_in_group = games_per_group
                workloads.append(
                    (
                        games_in_group, max_parallel_games, map_size, agent_1, agent_2, evaluation, False,
                        print_progress))
            for _ in range(int(self.workers_per_server / 2)):
                if leftovers > 0:
                    games_in_group = games_per_group + 1
                    leftovers -= 1
                else:
                    games_in_group = games_per_group
                workloads.append(
                    (games_in_group, max_parallel_games, map_size, agent_2, agent_1, evaluation, True, print_progress))
        return workloads


@ray.remote
class GameExecutorWorker:
    def __init__(self, id, progress_printer_actor, move_selection_strategy):
        self.id = id
        self.progress_printer = ParallelProgressPrinter(progress_printer_actor, self.id)
        self.move_selection_strategy = move_selection_strategy

    # this function is here to ensure that the agent fighting itself in self play is the same
    # object so searchtrees can be reused, because copying between processes creates two agents from the same one
    def play_self_play_games(self, game_count, max_parallel_games, map_size, agent, same_agent, evaluation,
                             agent_order_reversed, print_progress):
        return self.play_games_between_agents(game_count, max_parallel_games, map_size, agent, agent, evaluation,
                                              agent_order_reversed, print_progress)

    def play_games_between_agents(self, game_count, max_parallel_games, map_size, agent_1, agent_2, evaluation,
                                  agent_order_reversed, print_progress):
        if print_progress:
            progress_printer = self.progress_printer
        else:
            progress_printer = BaseProgressPrinter()
        if evaluation:
            training_sample_generator_class = PlaceholderTrainingSampleGenerator
        else:
            training_sample_generator_class = SymmetricTrainingSampleGenerator

        game_group = GameGroup(game_count, max_parallel_games, map_size, agent_1, agent_2,
                               training_sample_generator_class=training_sample_generator_class,
                               move_selection_strategy=self.move_selection_strategy,
                               reversed_agent_order=agent_order_reversed,
                               progress_printer=progress_printer)
        results = game_group.play_all_games()
        if isinstance(agent_1, MCTSAgent):
            agent_1.model.inference_server.worker_finished.remote()
        else:
            agent_2.model.inference_server.worker_finished.remote()
        return results
