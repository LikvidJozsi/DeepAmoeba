from AmoebaPlayGround.Agents.HandWrittenAgent import HandWrittenAgent, AnyFromHighestLevelSelection
from AmoebaPlayGround.Agents.MCTS.MCTSAgent import MCTSAgent
from AmoebaPlayGround.Agents.MCTS.MCTSTree import MCTSTree
from AmoebaPlayGround.Agents.TensorflowModels import ResNetLike
from AmoebaPlayGround.GameExecution.Multithreading.GameParallelizer import ParallelGameExecutor
from AmoebaPlayGround.GameExecution.SingleThreadGameExecutor import SingleThreadGameExecutor
from AmoebaPlayGround.Training.Logger import Statistics
from AmoebaPlayGround.Training.TrainingSampleGenerator import TrainingSampleCollection

map_size = (8, 8)

neural_network_config = {
    "map_size": map_size,
    "neural_network": {
        "general": {
            "training_batch_size": 1,
            "inference_batch_size": 1500,
            "training_dataset_max_size": 200000,
            "training_epochs": 1
        },
        "graph": {
            "first_convolution_size": [3, 3],
            "network_depth": 6,
            "dropout": 0.0,
            "reg": 5e-5,
            "learning_rate": 0.8e-3,
            "loss_weights": [1, 3]  # good rule of thumb is 1 for policy and log2(np.prod(board_size)) for value
        }
    }
}
config = {
    "mcts": {
        "tree_type": "MCTSTree",
        "search_count": 600,
        "max_intra_game_parallelism": 8,
        "exploration_rate": 1.2,
        "search_batch_size": 400,  # TODO refactor this config out, it shouldn't be a config, just function parameter
        "training_epochs": 4,
        "dirichlet_ratio": 0.0,
        "virtual_loss": 1
    }
}

model_1 = ResNetLike(config=neural_network_config)
model_1.load_model("2024-05-14_21-21-13")
agent_1 = MCTSAgent(model=model_1, config=config)

# model_2 = ResNetLike(config=neural_network_config)
# model_2.load_model("2022-11-15_22-21-52")
# agent_2 = MCTSAgent(model=model_2, config=config)

hand_written_agent = HandWrittenAgent()

game_executor_config = {
    "move_selection_strategy_type": "MoveSelectionStrategy",
    "inference_batch_size": 1500
}
# game_executor = ParallelGameExecutor(agent_1, hand_written_agent, game_executor_config)
game_executor = SingleThreadGameExecutor(None, None, game_executor_config)

sample_collection = TrainingSampleCollection()
self_play_statistics = Statistics()
game_count = 600

_, training_samples, statistics = game_executor.play_games_between_agents(game_count, agent_1, hand_written_agent,
                                                                          map_size,
                                                                          evaluation=True,
                                                                          print_progress=True)
print(
    f"agent_1 won: {statistics.games_won_by_player_1}, draw: {statistics.draw_games}, agent_2 won:{game_count - statistics.games_won_by_player_1 - statistics.draw_games}, avg game length: {statistics.get_average_game_length()}")
print(f"agent_1 score: {(statistics.games_won_by_player_1 + statistics.draw_games * 0.5) / game_count}")
