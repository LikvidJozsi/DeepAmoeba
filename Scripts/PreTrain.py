import pickle

import AmoebaPlayGround.Training.Evaluator as Evaluator
from AmoebaPlayGround.Agents.AmoebaAgent import RandomAgent
from AmoebaPlayGround.Agents.MCTS.MCTSAgent import MCTSAgent
from AmoebaPlayGround.Agents.TensorflowModels import ResNetLike
from AmoebaPlayGround.GameExecution.SingleThreadGameExecutor import SingleThreadGameExecutor
from AmoebaPlayGround.Training.Evaluator import EloEvaluator, ReferenceAgent
from AmoebaPlayGround.Training.Logger import ConsoleLogger
from AmoebaPlayGround.Training.TrainingSampleGenerator import TrainingDatasetGenerator

map_size = (8, 8)

neural_network_model = ResNetLike(training_batch_size=32,
                                  inference_batch_size=8000, training_dataset_max_size=400000)
neural_network_model.create_model(map_size, network_depth=6)

learning_agent = MCTSAgent(model=neural_network_model, search_count=300,
                           max_intra_game_parallelism=8)
game_executor = SingleThreadGameExecutor()

Evaluator.fix_reference_agents = [ReferenceAgent(name='random_agent', instance=RandomAgent(),
                                                 evaluation_match_count=10)]
evaluator = EloEvaluator(game_executor, map_size, puzzle_variation_count=10)
evaluator.evaluate_agent(learning_agent, ConsoleLogger())

dataset_size = 400000
with open("../Datasets/quickstart_dataset_8x8_300_searches.p", "rb") as file:
    dataset_generator = TrainingDatasetGenerator(pickle.load(file))
    inputs, output_policies, output_values = dataset_generator.get_dataset(dataset_size)
    output_policies = output_policies.reshape(output_policies.shape[0], -1)

evaluation_split_index = int(dataset_size * 4 / 5)

training_inputs = inputs[:evaluation_split_index]
training_output_policies = output_policies[:evaluation_split_index]
training_output_values = output_values[:evaluation_split_index]

evaluation_inputs = inputs[evaluation_split_index:]
evaluation_output_policies = output_policies[evaluation_split_index:]
evaluation_output_values = output_values[evaluation_split_index:]

for i in range(5):
    print("epoch " + str(i))

    neural_network_model.model.fit(x=training_inputs, y=[training_output_policies, training_output_values],
                                   validation_data=(
                                       evaluation_inputs, [evaluation_output_policies, evaluation_output_values]),
                                   epochs=1,
                                   shuffle=True,
                                   verbose=1, batch_size=32)
    evaluator.evaluate_agent(learning_agent, ConsoleLogger())

neural_network_model.save_model("pretrained_8x8")
