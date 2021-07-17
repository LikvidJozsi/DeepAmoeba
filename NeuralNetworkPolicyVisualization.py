import numpy as np

import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.AmoebaView import GraphicalView
from AmoebaPlayGround.BatchMCTSAgent import BatchMCTSAgent

Amoeba.map_size = (15, 15)

graphical_view = GraphicalView(Amoeba.map_size)
neural_agent = BatchMCTSAgent(search_count=1000, model_name="training_test", batch_size=200, map_size=Amoeba.map_size)

background_color = np.ones((15, 15), dtype=int) * 10
game = Amoeba.AmoebaGame(view=graphical_view)
game.play_game(graphical_view, graphical_view, neural_agent)
