from AmoebaPlayGround.Agents.MCTS.DictMCTSTree import DictMCTSTree
from AmoebaPlayGround.Agents.MCTS.MCTSAgent import MCTSAgent
from AmoebaPlayGround.Agents.MCTS.MCTSTree import MCTSTree
from AmoebaPlayGround.GameExecution.Multithreading.GameParallelizer import ParallelGameExecutor
from AmoebaPlayGround.Training.Logger import Statistics
from AmoebaPlayGround.Training.TrainingSampleGenerator import TrainingSampleCollection

agent_1 = MCTSAgent(search_count=600, search_batch_size=600,
                    max_intra_game_parallelism=8, tree_type=MCTSTree)

agent_2 = MCTSAgent(search_count=600, search_batch_size=600,
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
