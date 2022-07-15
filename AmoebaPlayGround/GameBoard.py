import hashlib
from enum import Enum

import numpy as np

EMPTY_SYMBOL = 0
X_SYMBOL = 1
O_SYMBOL = -1


class Player(Enum):
    X = 1
    O = -1
    NOBODY = 0

    def get_other_player(self):
        if self == Player.X:
            return Player.O
        else:
            return Player.X

    def get_symbol(self):
        if self == Player.X:
            return X_SYMBOL
        elif self == Player.O:
            return O_SYMBOL
        else:
            return None


class BoardIterator:
    def __init__(self, map):
        self.map = map
        self.row_index = 0

    def __next__(self):
        if self.map.get_number_of_rows() > self.row_index:
            row = self.map.get_row(self.row_index)
            self.row_index += 1
            return row
        raise StopIteration


class AmoebaBoard:
    def __init__(self, size=None, cells=None, occupied_cells=0):
        if cells is None:
            self.cells = np.zeros(size, dtype=np.uint8)
        else:
            self.cells = cells
        self.occupied_cells = occupied_cells

    def __iter__(self):
        return BoardIterator(self)

    def get_shape(self):
        return self.cells.shape

    def set(self, index, content):
        self.cells[index] = content
        self.occupied_cells += 1

    def get(self, index):
        return self.cells[index]

    def get_row(self, row_index):
        return self.cells[row_index, :]

    def get_number_of_rows(self):
        return self.cells.shape[0]

    def get_size(self):
        return np.prod(self.get_shape())

    def is_within_bounds(self, index):
        return 0 <= index[0] and index[0] < self.get_shape()[0] and 0 <= index[1] and index[1] < self.get_shape()[1]

    def reset(self):
        self.cells.fill(EMPTY_SYMBOL)

    def is_cell_empty(self, index):
        return self.cells[index] == EMPTY_SYMBOL

    def get_middle_of_map_index(self):
        middle_of_map_index = round(self.get_shape()[0] / 2), round(self.get_shape()[1] / 2)
        return middle_of_map_index

    def get_numeric_representation_for_player(self, player):
        map_copy = self.cells.copy()
        if player == Player.O:
            map_copy = map_copy * -1
        return map_copy

    def __hash__(self):
        return hash(hashlib.sha1(self.cells).hexdigest())

    def __eq__(self, other):
        return np.array_equal(self.cells, other.cells)

    def copy(self):
        copied_cells = self.cells.copy()
        return AmoebaBoard(self.get_shape(), copied_cells, self.occupied_cells)
