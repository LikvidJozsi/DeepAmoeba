import sys

import ray

from AmoebaPlayGround.Training.Logger import Statistics


def print_progress(progress, turn, time, extra_search_data, ongoing_games):
    barLength = 20
    status = "in progress"
    if progress >= 1:
        progress = 1
        status = "done\n"
    block = int(round(barLength * progress))
    text = "\r[{0}] {1:.1f}%, turn number: {2:.1f} ,games ongoing: {3}, turn time: {4:.4f}, {5},  status: {6}".format(
        "#" * block + "-" * (barLength - block), progress * 100, turn, ongoing_games, time, extra_search_data, status)
    sys.stdout.write(text)
    sys.stdout.flush()

class BaseProgressPrinter:
    def print_progress(self, progress, turn, time, extra_search_data, ongoing_games):
        pass


class SingleThreadedProgressPrinter(BaseProgressPrinter):

    def print_progress(self, progress, turn, time, extra_search_data, ongoing_games):
        print_progress(progress, turn, time, extra_search_data, ongoing_games)


class ParallelProgressPrinter(BaseProgressPrinter):
    def __init__(self, printer_actor, id):
        self.printer_actor = printer_actor
        self.id = id

    def print_progress(self, progress, turn, time, extra_search_data, ongoing_games):
        self.printer_actor.turn_completed.remote(self.id, progress, turn, time, extra_search_data, ongoing_games)


@ray.remote
class ParallelProgressTrackerActor:
    def __init__(self, worker_count):
        self.worker_count = worker_count
        self.reset()

    def reset(self):
        self.turns = dict()
        self.turn_statistics = dict()
        self.turn_times = dict()
        self.game_completed_fractions = dict()
        self.ongoing_games = dict()

    def turn_completed(self, worker_id, game_completed_fraction, turn, time, turn_statistics, ongoing_games):
        self.turns[worker_id] = turn
        self.turn_times[worker_id] = time
        self.game_completed_fractions[worker_id] = game_completed_fraction
        self.turn_statistics[worker_id] = turn_statistics
        self.ongoing_games[worker_id] = ongoing_games

    def have_all_games_finished(self):
        return len(self.ongoing_games.values()) > 0 and self.get_sum_of_ongoing_games() == 0

    def get_combined_metrics(self):
        average_turns = self.get_average_turns()
        turn_time = self.get_turn_time()
        completed_fraction = self.get_average_completed_fraction()
        combined_statistics = self.get_combined_statistics()
        sum_ongoing_games = self.get_sum_of_ongoing_games()
        return completed_fraction, average_turns, turn_time, combined_statistics, sum_ongoing_games

    def get_combined_statistics(self):
        combined_statistics = Statistics()
        for statistics in self.turn_statistics.values():
            combined_statistics.merge_statistics(statistics)
        return combined_statistics

    def get_turn_time(self):
        sum_turn_times = 0
        for turn_time in self.turn_times.values():
            sum_turn_times += turn_time
        return sum_turn_times / self.worker_count / self.worker_count

    def get_average_completed_fraction(self):
        sum_fractions = 0
        for turn in self.game_completed_fractions.values():
            sum_fractions += turn
        return sum_fractions / self.worker_count

    def get_average_turns(self):
        sum_turns = 0
        for turn in self.turns.values():
            sum_turns += turn
        return sum_turns / self.worker_count

    def get_sum_of_ongoing_games(self):
        sum_games = 0
        for game_count in self.ongoing_games.values():
            sum_games += game_count
        return sum_games

    def print_progress(self, progress, turn, time, extra_search_data, ongoing_games):
        barLength = 20
        status = "in progress"
        if progress >= 1:
            progress = 1
            status = "done\n"
        block = int(round(barLength * progress))
        text = "\r[{0}] {1:.1f}%, turn number: {2:.1f} ,games ongoing: {3}, turn time: {4:.4f}, {5},  status: {6}".format(
            "#" * block + "-" * (barLength - block), progress * 100, turn, ongoing_games, time, extra_search_data, status)
        sys.stdout.write(text)
        sys.stdout.flush()
