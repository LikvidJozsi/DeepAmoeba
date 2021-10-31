import pickle

from AmoebaPlayGround import Amoeba
from AmoebaPlayGround.Agents.MCTS.TreeMCTSAgent import TreeMCTSAgent
from AmoebaPlayGround.GameExecution.GameParallelizer import ParallelGameExecutor
from AmoebaPlayGround.Training.TrainingSampleGenerator import TrainingSampleCollection

Amoeba.map_size = (15, 15)

neural_agent = TreeMCTSAgent(search_count=500, load_latest_model=True, batch_size=200, map_size=Amoeba.map_size)

game_executor = ParallelGameExecutor(neural_agent, neural_agent, 10)
# game_executor = SingleThreadGameExecutor()
combined_training_samples = TrainingSampleCollection()
for i in range(1):
    _, training_samples, _ = game_executor.play_games_between_agents(480, neural_agent, neural_agent, evaluation=False,
                                                                     print_progress=True)
    combined_training_samples.extend(training_samples)

with open("Datasets/evaluation_dataset.p", 'wb') as file:
    pickle.dump(combined_training_samples, file)
