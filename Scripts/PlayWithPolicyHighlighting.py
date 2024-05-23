import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.Agents.MCTS.MCTSAgent import MCTSAgent
from AmoebaPlayGround.Agents.TensorflowModels import ResNetLike
from AmoebaPlayGround.AmoebaView import GraphicalView
from AmoebaPlayGround.Agents.HandWrittenAgent import HandWrittenAgent

map_size = (10, 10)


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


neural_network_model = ResNetLike(neural_network_config)
neural_network_model.load_model("2024-05-16_08-27-48")
graphical_view = GraphicalView(map_size)

config = {
    "mcts": {
        "tree_type": "MCTSTree",
        "search_count": 600,
        "max_intra_game_parallelism": 8,
        "exploration_rate": 1.4,
        "search_batch_size": 400,  # TODO refactor this config out, it shouldn't be a config, just function parameter
        "training_epochs": 4,
        "dirichlet_ratio": 0.1,
        "virtual_loss": 1
    }
}

neural_agent = MCTSAgent(model=neural_network_model, config=config)
hand_written_agent = HandWrittenAgent()
game = Amoeba.AmoebaGame(view=graphical_view, map_size=map_size)
game.play_game(neural_agent, graphical_view, neural_agent)
