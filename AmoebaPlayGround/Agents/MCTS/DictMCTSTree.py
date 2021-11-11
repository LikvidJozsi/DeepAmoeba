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

    def get_node_count(self):
        return len(self.tree)


class DictMCTSNode(MCTSNode):
    def __init__(self, board_state: AmoebaBoard, has_game_ended=False, parent=None, move_from_parent=None, turn=1,
                 **kwargs):
        super().__init__(board_state=board_state, has_game_ended=has_game_ended, **kwargs)
        self.turn = turn
        self.parents = dict()
        if parent is not None:
            self.parents[parent] = move_from_parent

    def add_parent(self, parent, move):
        self.parents[parent] = move

    def policy_calculation_started(self):
        for parent, move in self.parents.items():
            parent.child_policy_calculation_started(move)

    def policy_calculation_ended(self):
        for parent, move in self.parents.items():
            parent.child_policy_calculation_ended(move)

    def child_policy_calculation_started(self, move_of_child):
        self.invalid_moves[move_of_child] = True

    def child_policy_calculation_ended(self, move_of_child):
        self.invalid_moves[move_of_child] = False


class DictMCTSRootNode(DictMCTSNode, MCTSRootNode):
    def __init__(self, board_state: AmoebaBoard, has_game_ended=False, turn=1, eps=0.25):
        super().__init__(board_state=board_state, has_game_ended=has_game_ended, eps=eps, turn=turn)

    def policy_calculation_started(self):
        self.pending_policy_calculation = True

    def policy_calculation_ended(self):
        self.pending_policy_calculation = False


class DictMCTSTree(BaseMCTSTree):
    def __init__(self, current_turn=1):
        self.tree_levels = []
        self.current_turn = current_turn

    def get_the_node_of_move(self, search_node, move, player):
        new_move_tree_level = search_node.turn + 1
        new_board_state = search_node.get_board_state_after_move(move, player)
        node_of_new_board_state = self.get_search_node_of_board_state(new_board_state, new_move_tree_level, move,
                                                                      search_node)
        return node_of_new_board_state

    def get_existing_search_node(self, board_state, turn) -> MCTSNode:
        tree_level = self.get_tree_level(turn)
        return tree_level.get(board_state)

    def get_search_node_of_board_state(self, board_state, turn, move, parent_node):
        tree_level = self.get_tree_level(turn)
        search_node = tree_level.get(board_state)
        if search_node is not None:
            search_node.add_parent(parent_node, move)
            return search_node
        else:
            new_node = DictMCTSNode(board_state, turn=turn, parent=parent_node, move_from_parent=move)
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

    def get_node_count(self):
        sum_node_count = 0
        for level in self.tree_levels:
            sum_node_count += len(level)
        return sum_node_count
