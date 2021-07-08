import pickle

from AmoebaPlayGround import Amoeba
from AmoebaPlayGround.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.GameParallelizer import ParallelGameExecutor

Amoeba.map_size = (15, 15)

neural_agent = BatchMCTSAgent(search_count=500, load_latest_model=False, batch_size=200, map_size=Amoeba.map_size)

game_executor = ParallelGameExecutor(neural_agent, neural_agent, 6)
# game_executor = SingleThreadGameExecutor()
_, training_samples, _ = game_executor.play_games_between_agents(240, neural_agent, neural_agent, evaluation=False,
                                                                 print_progress=True)

with open("test_dataset.p", 'wb') as file:
    pickle.dump(training_samples, file)
