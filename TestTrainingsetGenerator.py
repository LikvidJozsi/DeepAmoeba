import pickle

from AmoebaPlayGround import Amoeba
from AmoebaPlayGround.Agents.MCTS.MCTSAgent import MCTSAgent
from AmoebaPlayGround.Agents.MCTS.MCTSTree import MCTSTree
from AmoebaPlayGround.Agents.TensorflowModels import ResNetLike
from AmoebaPlayGround.GameExecution.Multithreading.GameParallelizer import ParallelGameExecutor
from AmoebaPlayGround.Training.Logger import Statistics
from AmoebaPlayGround.Training.TrainingSampleGenerator import TrainingSampleCollection

Amoeba.map_size = (15, 15)
neural_network_model = ResNetLike(Amoeba.map_size, network_depth=6, training_batch_size=32,
                                  inference_batch_size=6000)
neural_network_model.create_model()
neural_agent = MCTSAgent(model=neural_network_model, search_batch_size=300,
                         map_size=Amoeba.map_size,
                         max_intra_game_parallelism=8, tree_type=MCTSTree)
game_executor = ParallelGameExecutor(neural_agent, neural_agent, worker_count=18, inference_server_count=3)
# game_executor = SingleThreadGameExecutor()

sample_collection = TrainingSampleCollection()
self_play_statistics = Statistics()
game_count = 4500
# game_count = 120
for i in range(1):
    _, training_samples, staticsitcs = game_executor.play_games_between_agents(game_count, neural_agent, neural_agent,
                                                                               evaluation=False,
                                                                               print_progress=True)
    sample_collection.extend(training_samples)
    self_play_statistics.merge_statistics(staticsitcs)

with open("Datasets/asd_dataset.p", 'wb') as file:
    pickle.dump(sample_collection, file)

with open("Datasets/asd_dataset_statistics.p", 'wb') as file:
    pickle.dump(self_play_statistics, file)
