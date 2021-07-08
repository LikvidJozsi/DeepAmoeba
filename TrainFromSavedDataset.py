import pickle

from AmoebaPlayGround import Amoeba
from AmoebaPlayGround.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.Logger import Logger
from AmoebaPlayGround.NetworkModels import ResNetLike
from AmoebaPlayGround.Puzzles import PuzzleEvaluator

Amoeba.map_size = (15, 15)

with open("test_dataset.p", 'rb') as file:
    dataset = pickle.load(file)
    neural_agent = BatchMCTSAgent(search_count=500, load_latest_model=False, batch_size=200, map_size=Amoeba.map_size,
                                  model_type=ResNetLike(6))

    neural_agent.print_model_summary()
    neural_agent.train(dataset)
    puzzle_evaluator = PuzzleEvaluator(25)
    puzzle_evaluator.evaluate_agent(neural_agent, Logger())
