import os
from os.path import exists

import numpy as np
from typing.io import IO

logs_folder = 'Logs/'
field_separator = ','


class Statistics:
    def __init__(self):
        self.max_search_depth = 0
        self.aggregate_search_depth = 0
        self.aggregate_search_tree_size = 0
        self.searches_done = 0
        self.aggregate_game_length = 0
        self.step_count = 0
        self.game_count = 0
        self.games_won_by_player_1 = 0
        self.games_won_by_player_2 = 0
        self.games_won_by_first_player = 0
        self.draw_games = 0
        self.aggregate_fraction_not_visited = 0
        self.aggregate_fraction_visited_once = 0
        self.aggregate_fraction_visited_twice = 0
        self.aggregate_fraction_visited_at_least_10_times = 0

    def get_average_game_length(self):
        if self.game_count > 0:
            return self.aggregate_game_length / self.game_count
        else:
            return 0

    def log(self, logger):
        logger.log("max_search_depth", self.max_search_depth)
        avg_search_depth = self.aggregate_search_depth / self.searches_done if self.searches_done > 0 else 0
        logger.log("avg_search_depth", avg_search_depth)
        avg_game_length = self.aggregate_game_length / self.game_count if self.game_count > 0 else 0
        logger.log("avg_game_length", avg_game_length)
        avg_node_count = self.aggregate_search_tree_size / self.step_count if self.step_count > 0 else 0
        logger.log("avg_node_count", avg_node_count)
        avg_fraction_not_visited = self.aggregate_fraction_not_visited / self.step_count if self.step_count > 0 else 0
        logger.log("avg_fraction_not_visited", avg_fraction_not_visited)
        avg_fraction_visited_once = self.aggregate_fraction_visited_once / self.step_count if self.step_count > 0 else 0
        logger.log("avg_fraction_visited_once", avg_fraction_visited_once)
        avg_fraction_visited_twice = self.aggregate_fraction_visited_twice / self.step_count if self.step_count > 0 else 0
        logger.log("avg_fraction_visited_twice", avg_fraction_visited_twice)
        avg_fraction_visited_at_least_10_times = self.aggregate_fraction_visited_at_least_10_times / self.step_count if self.step_count > 0 else 0
        logger.log("avg_fraction_visited_at_least_10_times", avg_fraction_visited_at_least_10_times)
        fraction_won_by_player_1 = self.games_won_by_first_player / self.game_count if self.game_count > 0 else 0
        logger.log("fraction_won_by_player_1", fraction_won_by_player_1)
        fraction_draw = self.draw_games / self.game_count if self.game_count > 0 else 0
        logger.log("fraction_draw", fraction_draw)

    def merge_statistics(self, other_statistics):
        self.max_search_depth = max(self.max_search_depth, other_statistics.max_search_depth)
        self.aggregate_search_depth += other_statistics.aggregate_search_depth
        self.searches_done += other_statistics.searches_done
        self.aggregate_game_length += other_statistics.aggregate_game_length
        self.step_count += other_statistics.step_count
        self.game_count += other_statistics.game_count
        self.games_won_by_player_1 += other_statistics.games_won_by_player_1
        self.games_won_by_player_2 += other_statistics.games_won_by_player_2
        self.games_won_by_first_player += other_statistics.games_won_by_first_player
        self.aggregate_search_tree_size += other_statistics.aggregate_search_tree_size
        self.draw_games += other_statistics.draw_games
        self.aggregate_fraction_not_visited += other_statistics.aggregate_fraction_not_visited
        self.aggregate_fraction_visited_once += other_statistics.aggregate_fraction_visited_once
        self.aggregate_fraction_visited_twice += other_statistics.aggregate_fraction_visited_twice
        self.aggregate_fraction_visited_at_least_10_times += other_statistics.aggregate_fraction_visited_at_least_10_times

    def add_win_statistics(self, games_won_by_player_1, games_won_by_player_2, draw_games, games_won_by_x):
        self.games_won_by_player_1 = games_won_by_player_1
        self.games_won_by_player_2 = games_won_by_player_2
        self.draw_games = draw_games
        self.games_won_by_first_player = games_won_by_x

    def add_searches(self, paths):
        new_searches = len(paths)
        for path in paths:
            path_length = len(path) + 1  # leaf node is not in the list
            self.aggregate_search_depth += path_length
            self.max_search_depth = max(self.max_search_depth, path_length)
        self.searches_done += new_searches

    def add_tree_statistics(self, search_position):
        game_count = len(search_position)
        self.step_count += game_count

        for position in search_position:
            self.aggregate_search_tree_size += position.search_tree.get_node_count()
            root_node_visit_counts = position.search_node.move_visited_counts
            invalid_moves = position.search_node.invalid_moves
            invalid_move_count = np.count_nonzero(invalid_moves)
            valid_move_count = np.prod(invalid_moves.shape) - invalid_move_count
            if valid_move_count == 0:
                print("waaaaaaaat")
            self.aggregate_fraction_not_visited += (np.count_nonzero(
                root_node_visit_counts == 0) - invalid_move_count) / valid_move_count
            self.aggregate_fraction_visited_once += np.count_nonzero(root_node_visit_counts == 1) / valid_move_count
            self.aggregate_fraction_visited_twice += np.count_nonzero(root_node_visit_counts == 2) / valid_move_count
            self.aggregate_fraction_visited_at_least_10_times += np.count_nonzero(
                root_node_visit_counts >= 10) / valid_move_count

    def __str__(self):
        if self.searches_done > 0:
            return "max_search_depth: {0}, avg_search_depth: {1:.2f}, avg_tree_size: {2:.2f}".format(
                self.max_search_depth,
                self.aggregate_search_depth / self.searches_done,
                self.aggregate_search_tree_size / self.step_count)
        else:
            return ""


class Logger:
    def new_episode(self):
        pass

    def log(self, key, message):
        pass


class FileLogger(Logger):
    def __init__(self, log_file_name):
        if log_file_name is None or log_file_name == "":
            raise Exception("Bad string received.")

        self.log_file_name = log_file_name
        self.log_file_path = self.log_file_name + ".csv"
        self.log_file_path = os.path.join(logs_folder, self.log_file_path)

        if exists(self.log_file_path):
            with open(self.log_file_path, "r") as log_file:
                self.headers = log_file.readline().split(field_separator)
        else:
            self.headers = []
        self.field_names = []
        self.field_values = []

    def log(self, key, value):
        self.field_names.append(key)
        self.field_values.append(str(value))

    def get_latest_agent_rating(self):
        with open(self.log_file_path, "r") as log_file:
            rating_field_index = self.headers.index("agent_rating")
            latest_rating = log_file.readlines()[-1].split(field_separator)[rating_field_index]
            return float(latest_rating)

    def get_log_episode_count(self):
        with open(self.log_file_path, "r") as log_file:
            return len(log_file.readlines()) - 1

    def new_episode(self):
        with open(self.log_file_path, mode="a", newline='') as log_file:
            if len(self.headers) == 0:
                self.headers = self.field_names
                self._write_line(log_file, self.headers)
            self._write_line(log_file, self.field_values)
            self.field_names = []
            self.field_values = []

    def validate_fields(self, headers, field_names):
        for field_name in field_names:
            if field_name not in headers:
                raise Exception("Field name not found among headers: " + field_name)

    def _write_line(self, file: IO, fields):
        fields_string = field_separator.join(fields)
        file.write(fields_string + "\n")


class ConsoleLogger(Logger):
    def log(self, key, message):
        print(key + ": " + str(message))
