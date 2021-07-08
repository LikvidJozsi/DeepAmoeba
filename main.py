import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.AmoebaAgent import RandomAgent
from AmoebaPlayGround.AmoebaTrainer import AmoebaTrainer
from AmoebaPlayGround.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.GameParallelizer import SingleThreadGameExecutor
from AmoebaPlayGround.HandWrittenAgent import HandWrittenAgent
from AmoebaPlayGround.Input import get_model_filename
from AmoebaPlayGround.Logger import FileLogger
from AmoebaPlayGround.NetworkModels import ResNetLike

file_name = get_model_filename()
Amoeba.map_size = (15, 15)
Amoeba.win_sequence_length = 5

# gui_agent = GraphicalView(Amoeba.map_size)
learning_agent = BatchMCTSAgent(load_latest_model=False, batch_size=200, search_count=500, map_size=Amoeba.map_size,
                                model_type=ResNetLike(6))
learning_agent.print_model_summary()
random_agent = RandomAgent()
hand_written_agent = HandWrittenAgent()

# evaluator.evaluate_against_agent(gui_agent,hand_written_agent)
# trainer = AmoebaTrainer(learning_agent, teaching_agents=[random_agent, hand_written_agent], self_play=False,
#                        reward_calculator=PolicyGradientsWithNegativeTeaching())
game_executor = SingleThreadGameExecutor()
trainer = AmoebaTrainer(learning_agent, teaching_agents=[], self_play=True, trainingset_size=200000,
                        game_executor=None, worker_count=6)
trainer.train(batch_size=240, num_episodes=30, model_save_file=file_name, logger=FileLogger(file_name))
