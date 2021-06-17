import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.Evaluator import EloEvaluator

Amoeba.map_size = (15, 15)
evaluator = EloEvaluator()

neural_agent_1 = BatchMCTSAgent(search_count=1000, load_latest_model=True, batch_size=200)
neural_agent_2 = BatchMCTSAgent(search_count=1000, load_latest_model=False, batch_size=200)
evaluator.set_reference_agent(neural_agent_2)
print(evaluator.evaluate_against_previous_version(neural_agent_1, neural_agent_2))
