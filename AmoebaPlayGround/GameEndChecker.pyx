cimport numpy as cnp
import cython

@cython.boundscheck(False)
def check_victory_condition(cnp.uint8_t[:,:]  game_board ,int[:] move,int win_sequence_length = 5):
    cdef int player_symbol = game_board[move[0],move[1]]
    player_won = (
            is_there_winning_line_in_direction(game_board, player_symbol, move[0],move[1],1, 0,win_sequence_length) or  # vertical
            is_there_winning_line_in_direction(game_board, player_symbol, move[0],move[1], 1, 1, win_sequence_length) or  # diagonal1
            is_there_winning_line_in_direction(game_board, player_symbol, move[0],move[1], 0, 1,win_sequence_length) or  # horizontal
            is_there_winning_line_in_reverse_diagonal(game_board, player_symbol, move[0],move[1],win_sequence_length))  # diagonal2
    is_draw = False
    return player_won



@cython.boundscheck(False)
cdef bint is_there_winning_line_in_direction(cnp.uint8_t[:,:]  game_board,int player_symbol,int move_x,int move_y,int dir_x, int dir_y,int win_sequence_length):
    # ....x....
    # only 4 places in each direction count in determining if the new move created a winning condition of
    # a five figure long line

    cdef int max_distance = win_sequence_length - 1
    cdef int max_x_neg_offset = get_maximum_negative_offset(move_x, dir_x, max_distance)
    cdef int max_y_neg_offset = get_maximum_negative_offset(move_y, dir_y, max_distance)
    cdef int max_neg_offset = min(max_x_neg_offset, max_y_neg_offset)

    board_size = game_board.shape
    cdef int max_x_pos_offset = get_maximum_positive_offset(move_x, dir_x, max_distance, board_size[0])
    cdef int max_y_pos_offset = get_maximum_positive_offset(move_y, dir_y, max_distance, board_size[1])
    cdef int max_pos_offset = min(max_x_pos_offset, max_y_pos_offset)

    cdef int line_length = 0
    cdef int offset
    cdef int x,y
    for offset in range(1, max_pos_offset + 1):
        # depending on the direction of the line being searched direction may be 0 meaning the coordinate does
        # not change on any iterations
        x = move_x + offset * dir_x
        y = move_y + offset * dir_y
        if game_board[x, y] == player_symbol:
            line_length += 1
        else:
            break

    for offset in range(-1, -max_neg_offset - 1, -1):
        # depending on the direction of the line being searched direction may be 0 meaning the coordinate does
        # not change on any iterations
        x = move_x + offset * dir_x
        y = move_y + offset * dir_y
        if game_board[x, y] == player_symbol:
            line_length += 1
        else:
            break

    return line_length >= win_sequence_length - 1

@cython.boundscheck(False)
cdef bint is_there_winning_line_in_reverse_diagonal(cnp.uint8_t[:,:] game_board,int player_symbol,int move_x,int move_y,int win_sequence_length):
    # in the reverse diagonal
    # ........x
    # .......x.
    # ......x..
    # .....x...
    # ....x....
    # one of the coordinates increases while to other decreases, so the boundary conditions get switched up

    board_size = game_board.shape
    cdef int max_distance = win_sequence_length - 1
    cdef int max_x_neg_offset = get_maximum_negative_offset(move_x, 1, max_distance)
    cdef int max_y_pos_offset = get_maximum_positive_offset(move_y, 1, max_distance, board_size[1])
    cdef int max_neg_offset = min(max_x_neg_offset, max_y_pos_offset)

    cdef int max_x_pos_offset = get_maximum_positive_offset(move_x, 1, max_distance, board_size[0])
    cdef int max_y_neg_offset = get_maximum_negative_offset(move_y, 1, max_distance)
    cdef int max_pos_offset = min(max_x_pos_offset, max_y_neg_offset)

    cdef int line_length = 0
    cdef int offset
    cdef int x, y
    for offset in range(1, max_pos_offset + 1):
        x = move_x + offset
        y = move_y - offset
        if game_board[x, y] == player_symbol:
            line_length += 1
        else:
            break

    for offset in range(-1, - max_neg_offset - 1, -1):
        x = move_x + offset
        y = move_y - offset
        if game_board[x, y] == player_symbol:
            line_length += 1
        else:
            break

    return line_length >= win_sequence_length - 1


cdef int get_maximum_negative_offset(int move_coordinate, int line_direction, int max_search_distance):
    return max_search_distance - max(max_search_distance * line_direction - move_coordinate, 0)

cdef int get_maximum_positive_offset(int move_coordinate,int line_direction,int max_search_distance,int map_length):
    return max_search_distance - max(max_search_distance * line_direction + move_coordinate - map_length + 1, 0)
