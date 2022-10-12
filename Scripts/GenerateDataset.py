import pickle

from AmoebaPlayGround.Agents.MCTS.MCTSAgent import MCTSAgent
from AmoebaPlayGround.Agents.MCTS.MCTSTree import MCTSTree
from AmoebaPlayGround.Agents.TensorflowModels import ConstantModel
from AmoebaPlayGround.GameExecution.Multithreading.GameParallelizer import ParallelGameExecutor

map_size = (8, 8)

neural_agent = MCTSAgent(model=ConstantModel(1500), search_count=300,
                         max_intra_game_parallelism=8,
                         tree_type=MCTSTree)

game_executor = ParallelGameExecutor(neural_agent, neural_agent, workers_per_inference_server=4,
                                     inference_server_count=3)

game_count = 4500

_, training_samples, statistics = game_executor.play_games_between_agents(game_count, neural_agent, neural_agent,
                                                                          map_size,
                                                                          evaluation=False,
                                                                          print_progress=True)

with open("../Datasets/quickstart_dataset_8x8_300_searches.p", 'wb') as file:
    pickle.dump(training_samples, file)

with open("../Datasets/quickstart_dataset_8x8_300_searches_statistics.p", 'wb') as file:
    pickle.dump(statistics, file)
