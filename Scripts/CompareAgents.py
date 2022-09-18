from AmoebaPlayGround.Agents.HandWrittenAgent import HandWrittenAgent, AnyFromHighestLevelSelection
from AmoebaPlayGround.Agents.MCTS.MCTSAgent import MCTSAgent
from AmoebaPlayGround.Agents.MCTS.MCTSTree import MCTSTree
from AmoebaPlayGround.Agents.TensorflowModels import ResNetLike
from AmoebaPlayGround.GameExecution.Multithreading.GameParallelizer import ParallelGameExecutor
from AmoebaPlayGround.Training.Logger import Statistics
from AmoebaPlayGround.Training.TrainingSampleGenerator import TrainingSampleCollection

map_size = (8, 8)
model_1 = ResNetLike(training_batch_size=32, inference_batch_size=8000)
model_1.create_model(map_size, network_depth=6)

agent_1 = MCTSAgent(model=model_1, search_count=600,
                    max_intra_game_parallelism=8,
                    tree_type=MCTSTree)

agent_2 = MCTSAgent(model=model_1.get_copy(), search_count=600,
                    max_intra_game_parallelism=8,
                    tree_type=MCTSTree)

hand_written_agent = HandWrittenAgent(AnyFromHighestLevelSelection())

game_executor = ParallelGameExecutor(agent_1, agent_2, 4)
# game_executor = SingleThreadGameExecutor()

sample_collection = TrainingSampleCollection()
self_play_statistics = Statistics()
game_count = 480

_, training_samples, statistics = game_executor.play_games_between_agents(game_count, agent_1, agent_2, map_size,
                                                                          evaluation=False,
                                                                          print_progress=True)
print(
    f"agent_1 won: {statistics.games_won_by_player_1}, draw: {statistics.draw_games}, agent_2 won:{game_count - statistics.games_won_by_player_1 - statistics.draw_games}, avg game length: {statistics.get_average_game_length()}")
print(f"agent_1 score: {(statistics.games_won_by_player_1 + statistics.draw_games * 0.5) / game_count}")
