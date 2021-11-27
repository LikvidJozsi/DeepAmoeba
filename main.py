import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.Agents.MCTS.TreeMCTSAgent import TreeMCTSAgent
from AmoebaPlayGround.Agents.TensorflowModels import ResNetLike
from AmoebaPlayGround.GameExecution.SingleThreadGameExecutor import SingleThreadGameExecutor
from AmoebaPlayGround.Training.AmoebaTrainer import AmoebaTrainer

Amoeba.map_size = (8, 8)
Amoeba.win_sequence_length = 5

# gui_agent = GraphicalView(Amoeba.map_size)
learning_agent = TreeMCTSAgent(load_latest_model=False, batch_size=1200, search_count=500, map_size=Amoeba.map_size,
                               model_type=ResNetLike(6),
                               max_intra_game_parallelism=8, training_dataset_max_size=600000)
learning_agent.print_model_summary()
# random_agent = RandomAgent()
# hand_written_agent = HandWrittenAgent()
exe = SingleThreadGameExecutor()
# evaluator.evaluate_against_agent(gui_agent,hand_written_agent)
# trainer = AmoebaTrainer(learning_agent, teaching_agents=[random_agent, hand_written_agent], self_play=False,
#                        reward_calculator=PolicyGradientsWithNegativeTeaching())
trainer = AmoebaTrainer(learning_agent, teaching_agents=[], self_play=True,
                        game_executor=None, worker_count=4,
                        training_sample_turn_cutoff_schedule=[(0, 10000), (1, 10000)],
                        resume_previous_training=False)
trainer.train(batch_size=720, batches_per_episode=2, num_episodes=15, use_quickstart_dataset=True)
