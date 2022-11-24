import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.Agents.MCTS.MCTSAgent import MCTSAgent
from AmoebaPlayGround.Agents.TensorflowModels import ResNetLike
from AmoebaPlayGround.AmoebaView import GraphicalView
from MCTSTree import MCTSTree

map_size = (8, 8)

neural_network_model = ResNetLike()
neural_network_model.load_model("2022-11-06_12-06-11_pretrained")

graphical_view = GraphicalView(map_size)
neural_agent = MCTSAgent(model=neural_network_model, search_count=600,
                         max_intra_game_parallelism=8,
                         tree_type=MCTSTree)

game = Amoeba.AmoebaGame(view=graphical_view, map_size=map_size)
game.play_game(neural_agent, graphical_view, neural_agent)
