import sys

from AmoebaPlayGround import Amoeba
from AmoebaPlayGround.Amoeba import AmoebaGame
from AmoebaPlayGround.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.GameBoard import Player
from AmoebaPlayGround.MCTSAgent import MCTSAgent

import numpy as np
Amoeba.map_size = (15, 15)
Amoeba.win_sequence_length = 5
sys.setrecursionlimit(5000)


batch_agent = BatchMCTSAgent(load_latest_model=False,batch_size=1)
batch_agent.save("regression_test")
print("miva")
simple_agent = MCTSAgent(model_name="regression_test")
print("mivaa")
game = AmoebaGame(None)
input = [game.map]
batch_probs = batch_agent.get_step(input,Player.O)
print("mivaaa")
simple_probs = simple_agent.get_step(input,Player.O)
print(batch_probs)
print(simple_probs)
print("the two outputs are equal: " + str(np.array_equal(batch_probs,simple_probs)))
