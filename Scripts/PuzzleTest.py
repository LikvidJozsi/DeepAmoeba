import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.Agents.MCTS.MCTSAgent import MCTSAgent
from AmoebaPlayGround.Agents.TensorflowModels import ResNetLike
from AmoebaPlayGround.GameExecution.MoveSelector import DistributionMoveSelector, \
    MaximalMoveSelector, EvaluationMoveSelectionStrategy
from AmoebaPlayGround.GameExecution.Multithreading.GameParallelizer import ParallelGameExecutor
from AmoebaPlayGround.Training.Logger import Logger
from AmoebaPlayGround.Training.Puzzles import PuzzleEvaluator

map_size = (8, 8)
Amoeba.win_sequence_length = 5

neural_network_model = ResNetLike(training_batch_size=32,
                                  inference_batch_size=8000, training_dataset_max_size=400000)
neural_network_model.load_model("2021-12-03_09-37-05")

learning_agent = MCTSAgent(model=neural_network_model, search_count=600,
                           max_intra_game_parallelism=8)

# evaluator.evaluate_against_agent(gui_agent,hand_written_agent)
# trainer = AmoebaTrainer(learning_agent, teaching_agents=[random_agent, hand_written_agent], self_play=False,
#                        reward_calculator=PolicyGradientsWithNegativeTeaching())
game_executor = ParallelGameExecutor(learning_agent, learning_agent, 6,
                                     EvaluationMoveSelectionStrategy(late_game_move_selector=MaximalMoveSelector(),
                                                                     late_game_start_turn=5,
                                                                     early_game_move_selector=DistributionMoveSelector(
                                                                         1 / 3)))

puzzle_evaluator = PuzzleEvaluator(20, map_size)
puzzle_evaluator.evaluate_agent(learning_agent, Logger())
