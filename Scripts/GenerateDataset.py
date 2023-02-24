import pickle

from AmoebaPlayGround.Agents.MCTS.MCTSAgent import MCTSAgent
from AmoebaPlayGround.Agents.TensorflowModels import ConstantModel
from AmoebaPlayGround.GameExecution.Multithreading.GameParallelizer import ParallelGameExecutor

map_size = (10, 10)

config = {
    "mcts": {
        "search_count": 300,
        "max_intra_game_parallelism": 8,
        "exploration_rate": 1.4,
        "search_batch_size": 400,  # TODO refactor this config out, it shouldn't be a config, just function parameter
        "training_epochs": 4,
        "dirichlet_ratio": 0.1,
        "virtual_loss": 1,
        "tree_type": "MCTSTree"  # MCTSTree or DictMCTSTree
    }
}
neural_agent = MCTSAgent(model=ConstantModel(), config=config)

game_executor_config = {
    "workers_per_inference_server": 4,
    "inference_server_count": 3,
    "inference_batch_size": 1500,
    "move_selection_strategy_type": "MoveSelectionStrategy"  # MoveSelectionStrategy or EvaluationMoveSelectionStrategy
}

game_executor = ParallelGameExecutor(neural_agent, neural_agent, game_executor_config)

game_count = 2000

_, training_samples, statistics = game_executor.play_games_between_agents(game_count, neural_agent, neural_agent,
                                                                          map_size,
                                                                          evaluation=False,
                                                                          print_progress=True)

with open("../Datasets/quickstart_dataset_10x10_300_searches.p", 'wb') as file:
    pickle.dump(training_samples, file)

with open("../Datasets/quickstart_dataset_10x10_300_searches_statistics.p", 'wb') as file:
    pickle.dump(statistics, file)
