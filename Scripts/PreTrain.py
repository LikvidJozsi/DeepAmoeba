import pickle
from collections import defaultdict
from datetime import datetime

import AmoebaPlayGround.Training.Evaluator as Evaluator
from AmoebaPlayGround.Agents.AmoebaAgent import RandomAgent
from AmoebaPlayGround.Agents.MCTS.MCTSAgent import MCTSAgent
from AmoebaPlayGround.Agents.TensorflowModels import ResNetLike
from AmoebaPlayGround.GameExecution.SingleThreadGameExecutor import SingleThreadGameExecutor
from AmoebaPlayGround.Training.Evaluator import EloEvaluator, ReferenceAgent
from AmoebaPlayGround.Training.Logger import ConsoleLogger
from AmoebaPlayGround.Training.TrainingSampleGenerator import TrainingDatasetGenerator

map_size = (12, 12)

training_name = None


class PreTrainingLogger(ConsoleLogger):
    def __init__(self):
        self.metrics = defaultdict(lambda: [])

    def log(self, key, message):
        super().log(key, message)
        self.metrics[key].append(message)


neural_network_config = {
    "map_size": map_size,
    "neural_network": {
        "general": {
            "training_batch_size": 1,
            "inference_batch_size": 1500,
            "training_dataset_max_size": 200000,
            "training_epochs": 1
        },
        "graph": {
            "first_convolution_size": [3, 3],
            "network_depth": 6,
            "dropout": 0.0,
            "reg": 5e-5,
            "learning_rate": 0.8e-3,
            "weights_file": "2022-11-15_22-15-32_pretrained",
            "loss_weights": [1, 3]  # good rule of thumb is 1 for policy and log2(np.prod(board_size)) for value
        }
    }
}

config = {
    "mcts": {
        "tree_type": "MCTSTree",
        "search_count": 600,
        "max_intra_game_parallelism": 8,
        "exploration_rate": 1.4,
        "search_batch_size": 400,  # TODO refactor this config out, it shouldn't be a config, just function parameter
        "training_epochs": 4,
        "dirichlet_ratio": 0.1,
        "virtual_loss": 1
    }
}

game_executor_config = {
    "move_selection_strategy_type": "MoveSelectionStrategy",
    "inference_batch_size": 1500
}

neural_network_model = ResNetLike(config=neural_network_config)
neural_network_model.create_model()

learning_agent = MCTSAgent(model=neural_network_model, config=config)
game_executor = SingleThreadGameExecutor("placeholder", "placeholder", game_executor_config)

Evaluator.fix_reference_agents = [ReferenceAgent(name='random_agent', instance=RandomAgent(),
                                                 evaluation_match_count=50)]
evaluator = EloEvaluator(game_executor, map_size, puzzle_variation_count=10)
logger = PreTrainingLogger()
evaluator.evaluate_agent(learning_agent, logger)

dataset_size = 200000
with open("../Datasets/quickstart_dataset_12x12_400_searches.p", "rb") as file:
    dataset_generator = TrainingDatasetGenerator(pickle.load(file))
    inputs, output_policies, output_values = dataset_generator.get_dataset(dataset_size)
    output_policies = output_policies.reshape(output_policies.shape[0], -1)

print(len(inputs))
evaluation_split_index = int(len(inputs) * 4 / 5)

training_inputs = inputs[:evaluation_split_index]
training_output_policies = output_policies[:evaluation_split_index]
training_output_values = output_values[:evaluation_split_index]

evaluation_inputs = inputs[evaluation_split_index:]
evaluation_output_policies = output_policies[evaluation_split_index:]
evaluation_output_values = output_values[evaluation_split_index:]

epochs = 4
for i in range(epochs):
    print("epoch " + str(i))

    training_result = neural_network_model.model.fit(x=training_inputs,
                                                     y=[training_output_policies, training_output_values],
                                                     validation_data=(
                                                         evaluation_inputs,
                                                         [evaluation_output_policies, evaluation_output_values]),
                                                     epochs=1,
                                                     shuffle=True,
                                                     verbose=1, batch_size=8)
    evaluator.evaluate_agent(learning_agent, logger)
    history = training_result.history
    logger.log("combined_loss", history["loss"][0])
    logger.log("policy_loss", history["policy_loss"][0])
    logger.log("value_loss", history["value_loss"][0])
    logger.log("validation_combined_loss", history["val_loss"][0])
    logger.log("validation_policy_loss", history["val_policy_loss"][0])
    logger.log("validation_value_loss", history["val_value_loss"][0])

if training_name is None:
    date_time = datetime.now()
    training_name = date_time.strftime("%Y-%m-%d_%H-%M-%S") + "_pretrained"

neural_network_model.save_model(training_name)
with open("../PreTrainingLogs/" + training_name + ".p", 'wb') as file:
    pickle.dump(dict(logger.metrics), file)
