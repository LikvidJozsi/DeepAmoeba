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

map_size = (8, 8)

training_name = None


class PreTrainingLogger(ConsoleLogger):
    def __init__(self):
        self.metrics = defaultdict(lambda: [])

    def log(self, key, message):
        super().log(key, message)
        self.metrics[key].append(message)


neural_network_model = ResNetLike(training_batch_size=8,
                                  inference_batch_size=8000)
neural_network_model.create_model(map_size, network_depth=6, reg=2 * 1e-5, learning_rate=2 * 1e-3)

learning_agent = MCTSAgent(model=neural_network_model, search_count=300,
                           max_intra_game_parallelism=8)
game_executor = SingleThreadGameExecutor()

Evaluator.fix_reference_agents = [ReferenceAgent(name='random_agent', instance=RandomAgent(),
                                                 evaluation_match_count=50)]
evaluator = EloEvaluator(game_executor, map_size, puzzle_variation_count=10)
logger = PreTrainingLogger()
evaluator.evaluate_agent(learning_agent, logger)

dataset_size = 200000
with open("../Datasets/quickstart_dataset_8x8_600_searches.p", "rb") as file:
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

epochs = 2
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
