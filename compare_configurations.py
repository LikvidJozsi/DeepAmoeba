from AmoebaPlayGround import Amoeba
from AmoebaPlayGround.Agents.MCTS.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.Agents.MCTS.DictMCTSTree import DictMCTSTree
from AmoebaPlayGround.Agents.MCTS.MCTSTree import MCTSTree
from AmoebaPlayGround.GameExecution.Multithreading.GameParallelizer import ParallelGameExecutor
from AmoebaPlayGround.Training.Logger import Statistics
from AmoebaPlayGround.Training.TrainingSampleGenerator import TrainingSampleCollection

agent_1 = BatchMCTSAgent(search_count=600, load_latest_model=True, inference_batch_size=600, map_size=Amoeba.map_size,
                         max_intra_game_parallelism=8, tree_type=MCTSTree)

agent_2 = BatchMCTSAgent(search_count=600, load_latest_model=True, inference_batch_size=600, map_size=Amoeba.map_size,
                         max_intra_game_parallelism=8, tree_type=DictMCTSTree)

game_executor = ParallelGameExecutor(agent_1, agent_2, 4)
# game_executor = SingleThreadGameExecutor()

sample_collection = TrainingSampleCollection()
self_play_statistics = Statistics()

_, training_samples, staticsitcs = game_executor.play_games_between_agents(480, agent_1, agent_2,
                                                                           evaluation=False,
                                                                           print_progress=True)
print(staticsitcs.games_won_by_player_1)
print(staticsitcs.draw_games)
print((staticsitcs.games_won_by_player_1 + 0.5 * staticsitcs.draw_games) / 480)
