import os

import ray
from ray.util import ActorPool

from AmoebaPlayGround import TrainingSampleGenerator, Amoeba
from AmoebaPlayGround.AmoebaAgent import PlaceholderAgent
from AmoebaPlayGround.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.GameGroup import GameGroup
from AmoebaPlayGround.Logger import Statistics
from AmoebaPlayGround.MoveSelector import DistributionMoveSelector
from AmoebaPlayGround.TrainingSampleGenerator import SymmetricTrainingSampleGenerator, \
    TrainingSampleCollection, PlaceholderTrainingSampleGenerator


class GameExecutor:

    def play_games_between_agents(self, game_count, agent_1, agent_2, evaluation=False):
        pass

    def merge_results(self, game_group_results):
        combined_training_samples = TrainingSampleCollection()
        combined_statistics = Statistics()
        combined_games = []
        for game_group_result in game_group_results:
            games, training_samples, statistics = game_group_result
            combined_games.extend(games)
            combined_training_samples.extend(training_samples)
            combined_statistics.merge_statistics(statistics)
        return combined_games, combined_training_samples, combined_statistics


class SingleThreadGameExecutor(GameExecutor):

    def __init__(self):
        pass

    def play_games_between_agents(self, game_count, agent_1, agent_2, evaluation=False):
        move_selector = DistributionMoveSelector()
        games_per_group = max(1, int(game_count / 2))
        if evaluation:
            training_sample_generator_class = PlaceholderTrainingSampleGenerator
        else:
            training_sample_generator_class = SymmetricTrainingSampleGenerator
        game_group_1 = GameGroup(games_per_group, agent_1, agent_2, None, log_progress=True,
                                 training_sample_generator_class=training_sample_generator_class,
                                 move_selector=move_selector, evaluation=evaluation)
        game_group_2 = GameGroup(games_per_group, agent_2, agent_1, None, log_progress=True,
                                 training_sample_generator_class=training_sample_generator_class,
                                 move_selector=move_selector, evaluation=evaluation,
                                 reversed_agent_order=True)

        group_1_results = game_group_1.play_all_games()
        group_2_results = game_group_2.play_all_games()
        return self.merge_results([group_1_results, group_2_results])


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
        for i in range(worker_count):
            worker = GameExecutorWorker.remote(learning_agent.get_weights(), reference_agent.get_weights(),
                                               learning_agent.map_size, learning_agent.search_count,
                                               learning_agent.batch_size)
            learning_agent.add_synchronized_copy(worker.set_learning_agent_weights)
            reference_agent.add_synchronized_copy(worker.set_reference_agent_weights)
            workers.append(worker)
        self.worker_pool = ActorPool(workers)
        self.worker_count = worker_count
        self.learning_agent = learning_agent
        self.reference_agent = reference_agent

    def play_games_between_agents(self, game_count, agent_1, agent_2, evaluation=False):
        agent_1 = self.replace_neural_agents_with_placeholders(agent_1)
        agent_2 = self.replace_neural_agents_with_placeholders(agent_2)
        game_groups = self.generate_workloads(agent_1, agent_2, game_count, evaluation)
        results = self.worker_pool.map(lambda worker, params: worker.play_games_between_agents.remote(*params),
                                       game_groups)
        return self.merge_results(results)

    def replace_neural_agents_with_placeholders(self, agent):
        if agent == self.learning_agent:
            agent = PlaceholderAgent("learning_agent")
        if agent == self.reference_agent:
            agent = PlaceholderAgent("reference_agent")
        return agent

    def generate_workloads(self, agent_1, agent_2, game_count, evaluation):
        workloads = []
        games_per_group = max(1, int(game_count / self.worker_count))
        if evaluation:
            training_sample_generator_class = TrainingSampleGenerator
        else:
            training_sample_generator_class = SymmetricTrainingSampleGenerator
        for _ in range(int(self.worker_count / 2)):
            workloads.append((games_per_group, agent_1, agent_2, training_sample_generator_class, evaluation, False,
                              Amoeba.map_size))
        for _ in range(int(self.worker_count / 2)):
            workloads.append((games_per_group, agent_2, agent_1, training_sample_generator_class, evaluation, True,
                              Amoeba.map_size))
        return workloads


@ray.remote
class GameExecutorWorker:
    def __init__(self, learning_agent_weights, reference_agent_weights, map_size, search_count=1000, batch_size=200):
        self.learning_agent = BatchMCTSAgent(search_count=search_count, load_latest_model=False, batch_size=batch_size,
                                             map_size=map_size)
        self.learning_agent.set_weights(learning_agent_weights)
        self.reference_agent = BatchMCTSAgent(search_count=search_count, load_latest_model=False, batch_size=batch_size,
                                              map_size=map_size)
        self.reference_agent.set_weights(reference_agent_weights)

    def set_learning_agent_weights(self, agent_1_weights):
        self.learning_agent.set_weights(agent_1_weights)

    def set_reference_agent_weights(self, agent_2_weights):
        self.reference_agent.set_weights(agent_2_weights)

    def play_games_between_agents(self, game_count, agent_1, agent_2, training_sample_generator_class, evaluation,
                                  agent_order_reversed, map_size):
        Amoeba.map_size = map_size
        agent_1 = self.replace_placeholder_agent(agent_1)
        agent_2 = self.replace_placeholder_agent(agent_2)
        move_selector = DistributionMoveSelector()
        game_group = GameGroup(game_count, agent_1, agent_2,
                               training_sample_generator_class=training_sample_generator_class,
                               move_selector=move_selector, evaluation=evaluation,
                               reversed_agent_order=agent_order_reversed)
        return game_group.play_all_games()

    def replace_placeholder_agent(self, agent):
        if agent.name == "learning_agent":
            return self.learning_agent
        if agent.name == "reference_agent":
            return self.reference_agent
        return agent
