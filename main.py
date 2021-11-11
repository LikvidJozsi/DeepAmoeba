import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.Agents.MCTS.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.Agents.MCTS.MCTSTree import MCTSTree
from AmoebaPlayGround.Agents.TensorflowModels import ResNetLike
from AmoebaPlayGround.GameExecution.SingleThreadGameExecutor import SingleThreadGameExecutor
from AmoebaPlayGround.Training.AmoebaTrainer import AmoebaTrainer
from AmoebaPlayGround.Training.Input import get_model_filename
from AmoebaPlayGround.Training.Logger import FileLogger

file_name = get_model_filename()
Amoeba.map_size = (15, 15)
Amoeba.win_sequence_length = 5

# gui_agent = GraphicalView(Amoeba.map_size)
learning_agent = BatchMCTSAgent(load_latest_model=False, batch_size=1400, search_count=500, map_size=Amoeba.map_size,
                                model_type=ResNetLike(6), tree_type=MCTSTree)
learning_agent.print_model_summary()
# random_agent = RandomAgent()
# hand_written_agent = HandWrittenAgent()
exe = SingleThreadGameExecutor()
# evaluator.evaluate_against_agent(gui_agent,hand_written_agent)
# trainer = AmoebaTrainer(learning_agent, teaching_agents=[random_agent, hand_written_agent], self_play=False,
#                        reward_calculator=PolicyGradientsWithNegativeTeaching())
trainer = AmoebaTrainer(learning_agent, teaching_agents=[], self_play=True, trainingset_size=600000,
                        game_executor=None, worker_count=4,
                        training_sample_turn_cutoff_schedule=[(0, 1), (1, 3), (3, 5), (6, 7), (10, 10000)],
                        resume_previous_training=False)
trainer.train(batch_size=480, batches_per_episode=2, num_episodes=15, model_save_file=file_name,
              logger=FileLogger(file_name))
