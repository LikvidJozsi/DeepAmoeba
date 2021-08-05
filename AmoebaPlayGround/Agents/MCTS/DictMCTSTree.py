from AmoebaPlayGround.Agents.MCTS.BaseMCTSTree import MCTSNode, MCTSRootNode, BaseMCTSTree
from AmoebaPlayGround.GameBoard import AmoebaBoard


class BasicDictMCTSTree(BaseMCTSTree):

    def __init__(self, current_turn=1):
        self.tree = dict()

    def get_the_node_of_move(self, search_node, move, player):
        new_board_state = search_node.get_board_state_after_move(move, player)
        search_node = self.tree.get(new_board_state)
        if search_node is not None:
            return search_node
        else:
            new_node = MCTSNode(new_board_state)
            if move is not None:
                new_node.set_game_ended(move)
            self.tree[new_board_state] = new_node
            return new_node

    def set_turn(self, turn):
        pass

    def get_root_node(self, game, eps):
        search_node = self.tree.get(game.map)
        if search_node is not None:
            root_node = MCTSRootNode(search_node.board_state, eps=eps)
            root_node.set_policy(search_node.neural_network_policy)
        else:
            root_node = MCTSRootNode(game.map.copy(), eps=eps)
        return root_node


class DictMCTSNode(MCTSNode):
    def __init__(self, board_state: AmoebaBoard, has_game_ended=False, turn=1):
        super().__init__(board_state, has_game_ended)
        self.turn = turn


class DictMCTSRootNode(MCTSRootNode):
    def __init__(self, board_state: AmoebaBoard, has_game_ended=False, turn=1, eps=0.25):
        super().__init__(board_state, has_game_ended, eps)
        self.turn = turn


class DictMCTSTree(BaseMCTSTree):
    def __init__(self, current_turn=1):
        self.tree_levels = []
        self.current_turn = current_turn

    def get_the_node_of_move(self, search_node, move, player):
        new_move_tree_level = search_node.turn + 1
        new_board_state = search_node.get_board_state_after_move(move, player)
        node_of_new_board_state = self.get_search_node_of_board_state(new_board_state, new_move_tree_level, move)
        return node_of_new_board_state

    def get_existing_search_node(self, board_state, turn) -> MCTSNode:
        tree_level = self.get_tree_level(turn)
        return tree_level.get(board_state)

    def get_search_node_of_board_state(self, board_state, turn, move):
        tree_level = self.get_tree_level(turn)
        search_node = tree_level.get(board_state)
        if search_node is not None:
            return search_node
        else:
            new_node = DictMCTSNode(board_state, turn=turn)
            new_node.set_game_ended(move)
            tree_level[board_state] = new_node
            return new_node

    def get_root_node(self, game, eps):
        game_board = game.map.copy()
        search_node = self.get_existing_search_node(game_board, game.num_steps)
        if search_node is not None:
            root_node = DictMCTSRootNode(search_node.board_state, turn=game.num_steps, eps=eps)
            root_node.set_policy(search_node.neural_network_policy)
        else:
            root_node = DictMCTSRootNode(game_board, turn=game.num_steps, eps=eps)
        return root_node

    def get_tree_level(self, turn):
        tree_level_array_index = turn - self.current_turn
        if tree_level_array_index < len(self.tree_levels):
            return self.tree_levels[tree_level_array_index]
        else:
            new_level = dict()
            self.tree_levels.append(new_level)
            return new_level

    def set_turn(self, new_turn):
        progress = new_turn - self.current_turn
        self.current_turn = new_turn
        self.tree_levels = self.tree_levels[progress:]
