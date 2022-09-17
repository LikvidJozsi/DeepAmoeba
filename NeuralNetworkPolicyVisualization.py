import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.Agents.MCTS.MCTSAgent import MCTSAgent
from AmoebaPlayGround.AmoebaView import GraphicalView

Amoeba.map_size = (8, 8)

graphical_view = GraphicalView(Amoeba.map_size)
neural_agent = MCTSAgent(search_count=1000, load_latest_model=False, model_name="2021-11-29_01-31-33",
                         search_batch_size=200, map_size=Amoeba.map_size)

game = Amoeba.AmoebaGame(view=graphical_view)
game.play_game(neural_agent, graphical_view, neural_agent)
