import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.Agents.HandWrittenAgent import HandWrittenAgent
from AmoebaPlayGround.Agents.MCTS.TreeMCTSAgent import TreeMCTSAgent
from AmoebaPlayGround.GameExecution.SingleThreadGameExecutor import SingleThreadGameExecutor
from AmoebaPlayGround.Training.Logger import Logger
from AmoebaPlayGround.Training.Puzzles import PuzzleEvaluator

Amoeba.map_size = (8, 8)
Amoeba.win_sequence_length = 5

# gui_agent = GraphicalView(Amoeba.map_size)
# learning_agent = TreeMCTSAgent(load_latest_model=False, batch_size=300, search_count=500, map_size=Amoeba.map_size)
learning_agent = TreeMCTSAgent(load_latest_model=True, batch_size=1000, search_count=2000, map_size=Amoeba.map_size)

# evaluator.evaluate_against_agent(gui_agent,hand_written_agent)
# trainer = AmoebaTrainer(learning_agent, teaching_agents=[random_agent, hand_written_agent], self_play=False,
#                        reward_calculator=PolicyGradientsWithNegativeTeaching())
game_executor = SingleThreadGameExecutor()
games, _, statistics = game_executor.play_games_between_agents(24, learning_agent, HandWrittenAgent(), evaluation=True,
                                                               print_progress=True)
print(
    f"mcts won: {statistics.games_won_by_player_1}, draw: {statistics.draw_games}, avg game length: {statistics.get_average_game_length()}")

puzzle_evaluator = PuzzleEvaluator(20)
puzzle_evaluator.evaluate_agent(learning_agent, Logger())
