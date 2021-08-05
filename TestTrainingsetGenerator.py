import pickle

from AmoebaPlayGround import Amoeba
from AmoebaPlayGround.Agents.MCTS.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.Agents.MCTS.MCTSTree import MCTSTree
from AmoebaPlayGround.GameExecution.GameParallelizer import SingleThreadGameExecutor
from AmoebaPlayGround.Training.TrainingSampleGenerator import TrainingSampleCollection

Amoeba.map_size = (15, 15)

neural_agent = BatchMCTSAgent(search_count=500, load_latest_model=False, batch_size=200, map_size=Amoeba.map_size,
                              tree_type=MCTSTree)

# game_executor = ParallelGameExecutor(neural_agent, neural_agent, 8)
game_executor = SingleThreadGameExecutor()
combined_training_samples = TrainingSampleCollection()
for i in range(8):
    _, training_samples, _ = game_executor.play_games_between_agents(60, neural_agent, neural_agent, evaluation=False,
                                                                     print_progress=True)
    combined_training_samples.extend(training_samples)

with open("large_dataset.p", 'wb') as file:
    pickle.dump(combined_training_samples, file)
