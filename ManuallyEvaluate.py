import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.Agents.AmoebaAgent import RandomAgent
from AmoebaPlayGround.Agents.HandWrittenAgent import HandWrittenAgent
from AmoebaPlayGround.Agents.MCTS.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.AmoebaView import GraphicalView
from AmoebaPlayGround.GameExecution.GameGroup import GameGroup

Amoeba.map_size = (8, 8)

graphical_view = GraphicalView(Amoeba.map_size)
hand_written_agent = HandWrittenAgent()
random_agent = RandomAgent()
neural_agent = BatchMCTSAgent(search_count=600, load_latest_model=False, model_name="2021-12-03_09-37-05",
                              inference_batch_size=400, map_size=Amoeba.map_size, max_intra_game_parallelism=8)
game = GameGroup(batch_size=1, x_agent=neural_agent, o_agent=graphical_view, view=graphical_view)
finished_games, training_samples, statistics, avg_time_per_turn_per_game = game.play_all_games()
