import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.Agents.AmoebaAgent import RandomAgent
from AmoebaPlayGround.Agents.MCTS.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.GameExecution.Multithreading.GameParallelizer import SingleThreadGameExecutor
from AmoebaPlayGround.Training.Evaluator import EloEvaluator

Amoeba.map_size = (15, 15)
game_executor = SingleThreadGameExecutor()
evaluator = EloEvaluator(game_executor)

neural_agent_1 = BatchMCTSAgent(search_count=500, load_latest_model=True, inference_batch_size=200,
                                map_size=Amoeba.map_size)
neural_agent_2 = BatchMCTSAgent(search_count=500, load_latest_model=False, inference_batch_size=200,
                                map_size=Amoeba.map_size)
random_agent = RandomAgent()
evaluator.set_reference_agent(random_agent)
print(evaluator.evaluate_against_previous_version(neural_agent_1, random_agent))
