import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.AmoebaAgent import RandomAgent
from AmoebaPlayGround.MCTSAgent import MCTSAgent
from AmoebaPlayGround.NeuralAgent import NeuralAgent
from AmoebaPlayGround.AmoebaView import GraphicalView
from AmoebaPlayGround.GameGroup import GameGroup
from AmoebaPlayGround.HandWrittenAgent import HandWrittenAgent

Amoeba.map_size = (15, 15)

graphical_view = GraphicalView(Amoeba.map_size)
hand_written_agent = HandWrittenAgent()
random_agent = RandomAgent()
# neural_agent = MCTSAgent(search_count=100)
game = GameGroup(batch_size=1, x_agent=graphical_view, o_agent=random_agent, view=graphical_view,log_progress=False)
game.play_all_games()