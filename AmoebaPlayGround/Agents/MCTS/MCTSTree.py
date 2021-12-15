import numpy as np

from AmoebaPlayGround.Agents.MCTS.BaseMCTSTree import MCTSNode, MCTSRootNode, BaseMCTSTree
from AmoebaPlayGround.Amoeba import AmoebaGame
from AmoebaPlayGround.GameBoard import AmoebaBoard


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

    def child_policy_calculation_started(self, move_of_child):
        self.invalid_moves[move_of_child] = True

    def child_policy_calculation_ended(self, move_of_child):
        self.invalid_moves[move_of_child] = False


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

    def get_node_count(self):
        # return self.get_subtree_node_count(self.root_node)
        return 0

    @staticmethod
    def policy_calculations_ended(nodes, selection_paths):
        pass
        '''for path, node in zip(selection_paths, nodes):
            if len(path) > 0:
                (node, move) = path[-1]
                node.child_policy_calculation_ended(move)'''

    @staticmethod
    def policy_calculation_started(node, selection_path):
        pass
        '''if len(selection_path) > 0:
            (previous_node, move) = selection_path[-1]
            previous_node.child_policy_calculation_started(move)'''

    def get_subtree_node_count(self, node):
        subtree_size = 0
        for index, child in np.ndenumerate(node.children):
            if child is not None:
                subtree_size += self.get_subtree_node_count(child)
        return subtree_size + 1
