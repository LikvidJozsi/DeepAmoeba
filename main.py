import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.AmoebaAgent import RandomAgent
from AmoebaPlayGround.AmoebaTrainer import AmoebaTrainer
from AmoebaPlayGround.AmoebaView import GraphicalView
from AmoebaPlayGround.Evaluator import EloEvaluator
from AmoebaPlayGround.HandWrittenAgent import HandWrittenAgent
from AmoebaPlayGround.Input import get_model_filename
from AmoebaPlayGround.Logger import AmoebaTrainingFileLogger
from AmoebaPlayGround.MCTSAgent import MCTSAgent
from AmoebaPlayGround.NeuralAgent import NeuralAgent

file_name = get_model_filename()

Amoeba.map_size = (15, 15)
Amoeba.win_sequence_length = 5

#gui_agent = GraphicalView(Amoeba.map_size)
learning_agent = MCTSAgent(load_latest_model=False)
learning_agent.print_model_summary()
random_agent = RandomAgent()
hand_written_agent = HandWrittenAgent()

#evaluator.evaluate_against_agent(gui_agent,hand_written_agent)
# trainer = AmoebaTrainer(learning_agent, teaching_agents=[random_agent, hand_written_agent], self_play=False,
#                        reward_calculator=PolicyGradientsWithNegativeTeaching())

trainer = AmoebaTrainer(learning_agent, teaching_agents=[], self_play=True)

trainer.train(batch_size=10, num_episodes=50, model_save_file=file_name, logger=AmoebaTrainingFileLogger(file_name))
