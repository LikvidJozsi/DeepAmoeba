import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.Agents.MCTS.MCTSAgent import MCTSAgent
from AmoebaPlayGround.Agents.MCTS.MCTSTree import MCTSTree
from AmoebaPlayGround.Agents.TensorflowModels import ResNetLike
from AmoebaPlayGround.Training.AmoebaTrainer import AmoebaTrainer

map_size = (15, 15)
Amoeba.win_sequence_length = 5

neural_network_model = ResNetLike(training_batch_size=32,
                                  inference_batch_size=8000, training_dataset_max_size=400000)
neural_network_model.create_model(map_size, network_depth=6)
neural_network_model.print_model_summary()

learning_agent = MCTSAgent(model=neural_network_model, search_count=600,
                           max_intra_game_parallelism=8,
                           tree_type=MCTSTree)

# exe = SingleThreadGameExecutor()
trainer = AmoebaTrainer(learning_agent, teaching_agents=[learning_agent], map_size=map_size,
                        game_executor=None, workers_per_inference_server=4,
                        training_sample_turn_cutoff_schedule=[(0, 10000), (1, 10000)],
                        resume_previous_training=False,
                        sample_episode_window_width_schedule=[(0, 1), (2, 2), (5, 4), (8, 6)])
trainer.train(batch_size=1440, num_episodes=30)
