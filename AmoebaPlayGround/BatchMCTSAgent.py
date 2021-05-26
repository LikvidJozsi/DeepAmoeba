from typing import List

from AmoebaPlayGround.GameBoard import AmoebaBoard
from AmoebaPlayGround.MCTSAgent import MCTSAgent


class BatchMCTSAgent(MCTSAgent):
    def get_step(self, game_boards: List[AmoebaBoard], player):
        pass

    def get_probability_distribution(self, search_node, player):
        pass

    def runSimulation(self, search_node, player, depth):
        pass

    def get_name(self):
        return 'BatchMCTSAgent'
