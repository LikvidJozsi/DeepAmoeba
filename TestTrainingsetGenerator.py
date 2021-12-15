import pickle

from AmoebaPlayGround import Amoeba
from AmoebaPlayGround.Agents.MCTS.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.GameExecution.GameParallelizer import ParallelGameExecutor
from AmoebaPlayGround.Training.Logger import Statistics
from AmoebaPlayGround.Training.TrainingSampleGenerator import TrainingSampleCollection

Amoeba.map_size = (15, 15)

neural_agent = BatchMCTSAgent(search_count=600, load_latest_model=False, batch_size=300, map_size=Amoeba.map_size,
                              max_intra_game_parallelism=8)

game_executor = ParallelGameExecutor(neural_agent, neural_agent, 6)
# game_executor = SingleThreadGameExecutor()

sample_collection = TrainingSampleCollection()
self_play_statistics = Statistics()
for i in range(2):
    _, training_samples, staticsitcs = game_executor.play_games_between_agents(720, neural_agent, neural_agent,
                                                                               evaluation=False,
                                                                               print_progress=True)
    sample_collection.extend(training_samples)
    self_play_statistics.merge_statistics(staticsitcs)

with open("Datasets/quickstart_dataset.p", 'wb') as file:
    pickle.dump(sample_collection, file)

with open("Datasets/quickstart_dataset_statistics.p", 'wb') as file:
    pickle.dump(self_play_statistics, file)
