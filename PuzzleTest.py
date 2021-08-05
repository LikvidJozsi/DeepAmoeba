import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.Input import get_model_filename
from AmoebaPlayGround.Logger import Logger
from AmoebaPlayGround.MCTS.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.MCTS.MCTSTree import MCTSTree
from AmoebaPlayGround.Puzzles import PuzzleEvaluator

file_name = get_model_filename()
Amoeba.map_size = (15, 15)
Amoeba.win_sequence_length = 5

# gui_agent = GraphicalView(Amoeba.map_size)
learning_agent = BatchMCTSAgent(load_latest_model=False, batch_size=300, search_count=500, map_size=Amoeba.map_size,
                                tree_type=MCTSTree)

# evaluator.evaluate_against_agent(gui_agent,hand_written_agent)
# trainer = AmoebaTrainer(learning_agent, teaching_agents=[random_agent, hand_written_agent], self_play=False,
#                        reward_calculator=PolicyGradientsWithNegativeTeaching())
puzzle_evaluator = PuzzleEvaluator(50)
puzzle_evaluator.evaluate_agent(learning_agent, Logger())
