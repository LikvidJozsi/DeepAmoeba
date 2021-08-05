import datetime
import pickle

import tensorflow as tf

from AmoebaPlayGround import Amoeba
from AmoebaPlayGround.Agents.MCTS.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.Agents.NetworkModels import ResNetLike
from AmoebaPlayGround.Training.Logger import Logger
from AmoebaPlayGround.Training.Puzzles import PuzzleEvaluator

Amoeba.map_size = (15, 15)

with open("large_dataset.p", 'rb') as train_file, open("evaluation_dataset.p", 'rb') as eval_file:
    train_dataset = pickle.load(train_file)
    evaluation_dataset = pickle.load(eval_file)

    log_dir = "TensorBoardLogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    neural_agent = BatchMCTSAgent(search_count=500, load_latest_model=False, batch_size=200, map_size=Amoeba.map_size,
                                  model_type=ResNetLike(6))

    neural_agent.print_model_summary()
    puzzle_evaluator = PuzzleEvaluator(25)
    neural_agent.train(train_dataset, validation_dataset=evaluation_dataset, epochs=10,
                       callbacks=[tensorboard_callback])
    puzzle_evaluator.evaluate_agent(neural_agent, Logger())
    neural_agent.save("training_test")
