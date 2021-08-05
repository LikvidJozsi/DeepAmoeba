import numpy as np

from AmoebaPlayGround.Amoeba import AmoebaGame
from AmoebaPlayGround.GameBoard import AmoebaBoard
from AmoebaPlayGround.MCTS.BaseMCTSTree import MCTSNode, MCTSRootNode, BaseMCTSTree


class TreeMCTSNode(MCTSNode):
    def __init__(self, board_state: AmoebaBoard, children=None, **kwargs):
        super().__init__(board_state=board_state, **kwargs)
        if children is None:
            self.children: np.ndarray[TreeMCTSNode] = np.full(board_state.get_shape(), None, dtype=TreeMCTSNode)
        else:
            self.children = children

    def get_the_node_of_move(self, move, player):
        child = self.children[move]
        if child is None:
            new_child = TreeMCTSNode(self.get_board_state_after_move(move, player))
            new_child.set_game_ended(move)
            self.children[move] = new_child
            return new_child
        else:
            return child

    def get_exsisting_node_of_move(self, move):
        return self.children[move]


class TreeMCTSRootNode(MCTSRootNode, TreeMCTSNode):
    def __init__(self, board_state: AmoebaBoard, has_game_ended=False, eps=0.25, children=None):
        super().__init__(board_state=board_state, has_game_ended=has_game_ended, eps=eps, children=children)


class MCTSTree(BaseMCTSTree):
    def __init__(self, current_turn=1):
        self.root_node = None
        self.root_node_turn = 0

    def get_the_node_of_move(self, search_node: TreeMCTSNode, move, player):
        return search_node.get_the_node_of_move(move, player)

    def get_root_node(self, game: AmoebaGame, eps):
        if self.root_node is None:
            self.root_node = TreeMCTSRootNode(game.map.copy(), eps=eps)
            self.root_node_turn = game.num_steps
            return self.root_node

        search_node = self.find_new_root_node_in_tree(game)

        if search_node is not None:
            self.root_node = TreeMCTSRootNode(search_node.board_state, eps=eps, children=search_node.children)
            self.root_node.set_policy(search_node.neural_network_policy)
        else:
            self.root_node = TreeMCTSRootNode(game.map.copy(), eps=eps)
        self.root_node_turn = game.num_steps
        return self.root_node

    def find_new_root_node_in_tree(self, game) -> TreeMCTSNode:
        moves = game.get_moves_since_turn(self.root_node_turn)

        current_node = self.root_node
        for move in moves:
            next_node_on_path = current_node.get_exsisting_node_of_move(move)
            if next_node_on_path is None:
                return None
            current_node = next_node_on_path
        return current_node

    def set_turn(self, new_turn):
        pass
