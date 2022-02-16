import datetime
import pickle

import tensorflow as tf

from AmoebaPlayGround import Amoeba
from AmoebaPlayGround.Agents.MCTS.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.Agents.TensorflowModels import ResNetLike
from AmoebaPlayGround.Training.Logger import Logger
from AmoebaPlayGround.Training.Puzzles import PuzzleEvaluator
from AmoebaPlayGround.Training.TrainingSampleGenerator import TrainingDatasetGenerator

Amoeba.map_size = (15, 15)

with open("Datasets/quickstart_dataset.p", 'rb') as train_file:
    train_dataset = pickle.load(train_file)
    train_dataset = TrainingDatasetGenerator(train_dataset)
    log_dir = "TensorBoardLogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    neural_agent = BatchMCTSAgent(search_count=600, load_latest_model=False, search_batch_size=400,
                                  map_size=Amoeba.map_size,
                                  model_type=ResNetLike(6))

    neural_agent.print_model_summary()
    puzzle_evaluator = PuzzleEvaluator(25)
    for i in range(10):
        neural_agent.train(train_dataset, epochs=1,
                           callbacks=[tensorboard_callback])
        print(i)
        puzzle_evaluator.evaluate_agent(neural_agent, Logger())
    neural_agent.save("training_test")
