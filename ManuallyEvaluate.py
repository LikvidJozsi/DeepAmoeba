import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.NeuralAgent import NeuralAgent
from AmoebaPlayGround.AmoebaView import GraphicalView
from AmoebaPlayGround.GameGroup import GameGroup
from AmoebaPlayGround.HandWrittenAgent import HandWrittenAgent

Amoeba.map_size = (15, 15)

graphical_view = GraphicalView(Amoeba.map_size)
hand_written_agent = HandWrittenAgent()
#neural_agent = NeuralAgent(model_name='2019-12-13_23-13-20')
game = GameGroup(batch_size=1, x_agent=hand_written_agent, o_agent=graphical_view, view=graphical_view)
game.play_all_games()
