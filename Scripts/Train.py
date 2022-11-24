import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.Agents.MCTS.MCTSAgent import MCTSAgent
from AmoebaPlayGround.Agents.TensorflowModels import ResNetLike
from AmoebaPlayGround.Training.AmoebaTrainer import AmoebaTrainer

map_size = (8, 8)
Amoeba.win_sequence_length = 5

neural_network_model = ResNetLike(training_batch_size=4,
                                  inference_batch_size=1500, training_dataset_max_size=200000,
                                  training_epochs=1)
neural_network_model.create_model(map_size, 6, reg=1 * 1e-5, learning_rate=8 * 1e-4)
neural_network_model.load_weights("2022-11-15_22-15-32_pretrained")
neural_network_model.print_model_summary()

learning_agent = MCTSAgent(model=neural_network_model, search_count=600,
                           max_intra_game_parallelism=8)

# exe = SingleThreadGameExecutor()
trainer = AmoebaTrainer(learning_agent, teaching_agents=[learning_agent], map_size=map_size,
                        game_executor=None, workers_per_inference_server=4,
                        inference_server_count=3,
                        training_sample_turn_cutoff_schedule=None,
                        resume_previous_training=True,
                        sample_episode_window_width_schedule=[(0, 1), (2, 2), (6, 3)])
trainer.train(batch_size=1500, num_episodes=30)
