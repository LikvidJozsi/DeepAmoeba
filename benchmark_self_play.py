import ray

from AmoebaPlayGround import Amoeba
from AmoebaPlayGround.Agents.MCTS.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.Agents.MCTS.MCTSTree import MCTSTree
from AmoebaPlayGround.GameExecution.GameParallelizer import ParallelGameExecutor
from AmoebaPlayGround.Training.Logger import Statistics
from AmoebaPlayGround.Training.TrainingSampleGenerator import TrainingSampleCollection

Amoeba.map_size = (8, 8)

neural_agent = BatchMCTSAgent(search_count=600, load_latest_model=True, batch_size=400, map_size=Amoeba.map_size,
                              max_intra_game_parallelism=8, tree_type=MCTSTree)

game_executor = ParallelGameExecutor(neural_agent, neural_agent, 8)
# game_executor = SingleThreadGameExecutor()

sample_collection = TrainingSampleCollection()
self_play_statistics = Statistics()

_, training_samples, staticsitcs = game_executor.play_games_between_agents(480, neural_agent, neural_agent,
                                                                           evaluation=False,
                                                                           print_progress=True)
print(staticsitcs.aggregate_search_tree_size / staticsitcs.step_count)

ray.shutdown()
