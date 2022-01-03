import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.Agents.MCTS.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.Agents.MCTS.MCTSTree import MCTSTree
from AmoebaPlayGround.Agents.TensorflowModels import ResNetLike
from AmoebaPlayGround.Training.AmoebaTrainer import AmoebaTrainer

Amoeba.map_size = (15, 15)
Amoeba.win_sequence_length = 5

neural_network_model = ResNetLike(Amoeba.map_size, network_depth=6, training_batch_size=32)
neural_network_model.create_model()
learning_agent = BatchMCTSAgent(model=neural_network_model, inference_batch_size=400, search_count=600,
                                map_size=Amoeba.map_size,
                                max_intra_game_parallelism=8, training_dataset_max_size=400000,
                                tree_type=MCTSTree)
neural_network_model.print_model_summary()

# exe = SingleThreadGameExecutor()
trainer = AmoebaTrainer(learning_agent, teaching_agents=[], self_play=True,
                        game_executor=None, worker_count=4,
                        training_sample_turn_cutoff_schedule=[(0, 10000), (1, 10000)],
                        resume_previous_training=False,
                        sample_episode_window_width_schedule=[(0, 1), (2, 2), (5, 4), (8, 6)])
trainer.train(batch_size=720, batches_per_episode=2, num_episodes=30, use_quickstart_dataset=True)
