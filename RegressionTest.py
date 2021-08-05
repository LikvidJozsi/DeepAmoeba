from AmoebaPlayGround import Amoeba
from AmoebaPlayGround.Agents.MCTS.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.Amoeba import AmoebaGame
from AmoebaPlayGround.GameBoard import Player
from AmoebaPlayGround.GameExecution.MoveSelector import MaximalMoveSelector

Amoeba.map_size = (15, 15)
Amoeba.win_sequence_length = 5

batch_agent = BatchMCTSAgent(load_latest_model=False, model_name="2021-06-21_20-22-27", batch_size=1)
batch_agent.save("regression_test")
# simple_agent = MCTSAgent(model_name="regression_test")
game = AmoebaGame(None)
batch_probs, _ = batch_agent.get_step([game], Player.O)
move_selector = MaximalMoveSelector()
selected_move = move_selector.select_move(batch_probs[0])
# simple_probs = simple_agent.get_step(input,Player.O)
print(game.map.cells)
print(selected_move)

'''print(simple_probs)
print("the two outputs are equal: " + str(np.array_equal(batch_probs,simple_probs)))'''
