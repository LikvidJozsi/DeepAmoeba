import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.Agents.MCTS.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.AmoebaView import GraphicalView

Amoeba.map_size = (8, 8)

graphical_view = GraphicalView(Amoeba.map_size)
neural_agent = BatchMCTSAgent(search_count=1000, load_latest_model=True, batch_size=200, map_size=Amoeba.map_size)

game = Amoeba.AmoebaGame(view=graphical_view)
game.play_game(graphical_view, graphical_view, neural_agent)
