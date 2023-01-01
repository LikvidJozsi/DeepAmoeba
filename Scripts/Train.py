import toml

import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.Agents.MCTS.MCTSAgent import MCTSAgent
from AmoebaPlayGround.Agents.TensorflowModels import ResNetLike
from AmoebaPlayGround.Training.AmoebaTrainer import AmoebaTrainer

map_size = (8, 8)
Amoeba.win_sequence_length = 5
config = None

with open("train_config.toml", "r") as config_file:
    config = toml.loads(config_file.read())

neural_network_model = ResNetLike(config)
neural_network_model.create_model()
neural_network_model.load_weights(config["neural_network"]["graph"]["weights_file"])  # TODO refactor this
neural_network_model.print_model_summary()

learning_agent = MCTSAgent(model=neural_network_model, config=config)

# exe = SingleThreadGameExecutor()
trainer = AmoebaTrainer(learning_agent, teaching_agents=[learning_agent], config=config)
trainer.train(batch_size=1500, num_episodes=30)
