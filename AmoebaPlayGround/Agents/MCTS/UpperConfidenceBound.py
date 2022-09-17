import numpy as np

from numba import njit


@njit(fastmath=True)
def get_best_ucb_node(sum_expected_move_rewards, neural_network_policy, move_visited_counts,
                      invalid_moves, exploration_rate, visited_count, board_shape):
    average_expected_reward = sum_expected_move_rewards / np.maximum(1, move_visited_counts)
    upper_confidence_bounds = average_expected_reward + exploration_rate * neural_network_policy * \
                              np.sqrt(visited_count + 1e-8) / (1 + move_visited_counts)

    valid_upper_confidence_bounds = np.where(invalid_moves, -np.inf, upper_confidence_bounds)
    best_move = np.argmax(valid_upper_confidence_bounds)
    index_1 = best_move // board_shape[0]
    index_2 = best_move - index_1 * board_shape[0]
    best_move = (index_1, index_2)
    if valid_upper_confidence_bounds[best_move] == -np.inf:
        return None
    return best_move
