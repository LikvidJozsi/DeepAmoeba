import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.Agents.HandWrittenAgent import HandWrittenAgent, AnyFromHighestLevelSelection
from AmoebaPlayGround.Agents.MCTS.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.GameExecution.MoveSelector import DistributionMoveSelector, \
    MaximalMoveSelector, EvaluationMoveSelectionStrategy
from AmoebaPlayGround.GameExecution.Multithreading.GameParallelizer import ParallelGameExecutor

Amoeba.map_size = (8, 8)
Amoeba.win_sequence_length = 5

# gui_agent = GraphicalView(Amoeba.map_size)
# learning_agent = TreeMCTSAgent(load_latest_model=False, batch_size=300, search_count=500, map_size=Amoeba.map_size)
learning_agent = BatchMCTSAgent(load_latest_model=False, model_name="2021-12-03_09-37-05", inference_batch_size=400,
                                search_count=150, map_size=Amoeba.map_size,
                                virtual_loss=1, exploration_rate=1.4, max_intra_game_parallelism=8)

# evaluator.evaluate_against_agent(gui_agent,hand_written_agent)
# trainer = AmoebaTrainer(learning_agent, teaching_agents=[random_agent, hand_written_agent], self_play=False,
#                        reward_calculator=PolicyGradientsWithNegativeTeaching())
game_executor = ParallelGameExecutor(learning_agent, learning_agent, 6,
                                     EvaluationMoveSelectionStrategy(late_game_move_selector=MaximalMoveSelector(),
                                                                     late_game_start_turn=5,
                                                                     early_game_move_selector=DistributionMoveSelector(
                                                                         1 / 3)))
game_count = 126
games, _, statistics = game_executor.play_games_between_agents(game_count, learning_agent,
                                                               HandWrittenAgent(AnyFromHighestLevelSelection()),
                                                               evaluation=True,
                                                               print_progress=True)
print(
    f"mcts won: {statistics.games_won_by_player_1}, draw: {statistics.draw_games}, mcts lost:{game_count - statistics.games_won_by_player_1 - statistics.draw_games}, avg game length: {statistics.get_average_game_length()}")
print(f"score: {(statistics.games_won_by_player_1 + statistics.draw_games * 0.5) / game_count}")
# puzzle_evaluator = PuzzleEvaluator(20)
# puzzle_evaluator.evaluate_agent(learning_agent, Logger())
