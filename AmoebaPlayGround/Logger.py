import os

from typing.io import IO

logs_folder = 'Logs/'
field_separator = ','


class Statistics:
    def __init__(self):
        self.max_search_depth = 0
        self.aggregate_search_depth = 0
        self.searches_done = 0
        self.aggregate_game_length = 0
        self.game_count = 0

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

    def merge_statistics(self, other_statistics):
        self.max_search_depth = max(self.max_search_depth, other_statistics.max_search_depth)
        self.aggregate_search_depth += other_statistics.aggregate_search_depth
        self.searches_done += other_statistics.searches_done
        self.aggregate_game_length += other_statistics.aggregate_game_length
        self.game_count += other_statistics.game_count

    def add_searches(self, paths):
        new_searches = len(paths)
        for path in paths:
            path_length = len(path) + 1  # leaf node is not in the list
            self.aggregate_search_depth += path_length
            self.max_search_depth = max(self.max_search_depth, path_length)
        self.searches_done += new_searches

    def __str__(self):
        if self.searches_done > 0:
            return "max_search_depth: {0}, avg_search_depth: {1:.2f}".format(self.max_search_depth,
                                                                             self.aggregate_search_depth / self.searches_done)
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
        self.headers = []
        self.field_names = []
        self.field_values = []

    def log(self, key, value):
        self.field_names.append(key)
        self.field_values.append(str(value))

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
