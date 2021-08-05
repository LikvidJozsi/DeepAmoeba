import sys

import ray

from AmoebaPlayGround.Training.Logger import Statistics


class BaseProgressPrinter:
    def print_progress(self, progress, turn, time, extra_search_data):
        pass


class SingleThreadedProgressPrinter(BaseProgressPrinter):

    def print_progress(self, progress, turn, time, extra_search_data):
        barLength = 20
        status = "in progress"
        if progress >= 1:
            progress = 1
            status = "done\r\n"
        block = int(round(barLength * progress))
        text = "\r[{0}] {1:.1f}%, turn number: {2} , turn time: {3:.4f}, {4}, status: {5}".format(
            "#" * block + "-" * (barLength - block), progress * 100, turn, time, extra_search_data, status)
        sys.stdout.write(text)
        sys.stdout.flush()


class ParallelProgressPrinter(BaseProgressPrinter):
    def __init__(self, printer_actor, id):
        self.printer_actor = printer_actor
        self.id = id

    def print_progress(self, progress, turn, time, extra_search_data):
        self.printer_actor.turn_completed.remote(self.id, progress, turn, time, extra_search_data)


@ray.remote
class ParallelProgressPrinterActor:
    def __init__(self, worker_count):
        self.worker_count = worker_count
        self.reset()

    def reset(self):
        self.turns = dict()
        self.turn_statistics = dict()
        self.turn_times = dict()
        self.game_completed_fractions = dict()

    def turn_completed(self, worker_id, game_completed_fraction, turn, time, turn_statistics):
        self.turns[worker_id] = turn
        self.turn_times[worker_id] = time
        self.game_completed_fractions[worker_id] = game_completed_fraction
        self.turn_statistics[worker_id] = turn_statistics
        metrics = self.combine_metrics()
        self.print_progress(*metrics)

    def combine_metrics(self):
        average_turns = self.get_average_turns()
        turn_time = self.get_turn_time()
        completed_fraction = self.get_average_completed_fraction()
        combined_statistics = self.get_combined_statistics()
        return completed_fraction, average_turns, turn_time, combined_statistics

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

    def print_progress(self, progress, turn, time, extra_search_data):
        barLength = 20
        status = "in progress"
        if progress >= 1:
            progress = 1
            status = "done\n"
        block = int(round(barLength * progress))
        text = "\r[{0}] {1:.1f}%, turn number: {2} , turn time: {3:.4f}, {4}, status: {5}".format(
            "#" * block + "-" * (barLength - block), progress * 100, turn, time, extra_search_data, status)
        sys.stdout.write(text)
        sys.stdout.flush()
