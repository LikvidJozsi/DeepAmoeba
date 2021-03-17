from enum import Enum

import numpy as np


class Symbol(Enum):
    X = 1
    O = -1
    EMPTY = 0


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
            return Symbol.X
        elif self == Player.O:
            return Symbol.O
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
    def __init__(self, size, perspective=Player.X):
        self.cells = np.empty(size, dtype=Symbol)
        self.cells.fill(Symbol.EMPTY)
        self.shape = size
        self.perspective = perspective

    def __iter__(self):
        return BoardIterator(self)

    def set(self, index, value):
        self.cells[index] = value

    def get(self, index):
        return self.cells[index]

    def get_row(self, row_index):
        return self.cells[row_index, :]

    def get_number_of_rows(self):
        return self.cells.shape[0]

    def get_size(self):
        return np.prod(self.shape)

    def is_within_bounds(self, index):
        return 0 <= index[0] and index[0] < self.shape[0] and 0 <= index[1] and index[1] < self.shape[1]

    def reset(self):
        self.cells.fill(Symbol.EMPTY)

    def is_cell_empty(self, index):
        return self.cells[index] == Symbol.EMPTY

    def get_middle_of_map_index(self):
        middle_of_map_index = round(self.shape[0] / 2), round(self.shape[1] / 2)
        return middle_of_map_index
