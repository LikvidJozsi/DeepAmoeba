from AmoebaPlayGround.MCTSAgent import MCTSNode


class MCTSTree:
    def __init__(self, current_turn=1):
        self.tree_levels = []
        self.current_turn = current_turn

    def get_the_node_of_move(self, search_node, move, player):
        new_move_tree_level = search_node.turn + 1
        new_board_state = search_node.get_board_state_after_move(move, player)
        node_of_new_board_state = self.get_search_node_of_board_state(new_board_state, new_move_tree_level, move)
        return node_of_new_board_state

    def get_existing_search_node(self, board_state, turn):
        tree_level = self.get_tree_level(turn)
        return tree_level.get(board_state)

    def get_search_node_of_board_state(self, board_state, turn, move=None):
        tree_level = self.get_tree_level(turn)
        search_node = tree_level.get(board_state)
        if search_node is not None:
            return search_node
        else:
            new_node = MCTSNode(board_state, turn=turn)
            if move is not None:
                new_node.set_game_ended(move)
            tree_level[board_state] = new_node
            return new_node

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
