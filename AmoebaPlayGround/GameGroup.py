import sys

from AmoebaPlayGround.Amoeba import AmoebaGame, Player
from AmoebaPlayGround.MoveSelector import MaximalMoveSelector
from AmoebaPlayGround.TrainingSampleGenerator import SymmetricTrainingSampleGenerator, TrainingSampleCollection


class GameGroup:
    def __init__(self, batch_size, x_agent, o_agent,
                 view=None, training_sample_generator_class=SymmetricTrainingSampleGenerator, log_progress=False,
                 move_selector=MaximalMoveSelector()):
        self.x_agent = x_agent
        self.o_agent = o_agent
        self.log_progress = log_progress
        self.games = []
        self.move_selector = move_selector
        self.training_sample_generators = []
        for index in range(batch_size):
            self.games.append(AmoebaGame(view))
            self.training_sample_generators.append(training_sample_generator_class())

    def play_all_games(self):
        finished_games = []
        number_of_games = len(self.games)
        training_samples = TrainingSampleCollection()
        while len(self.games) != 0:
            next_agent = self.get_next_agent(self.games[0])  # the same agent has its turn in every active game at the
            # same time, therfore getting the agent of any of them is enough
            maps = self.get_maps_of_games()
            action_probabilities = next_agent.get_step(maps, next_agent)
            for game, training_sample_generator, action_probabilities in zip(self.games,
                                                                             self.training_sample_generators,
                                                                             action_probabilities):
                action = self.move_selector.select_move(action_probabilities)
                game.step(action)
                training_sample_generator.add_move(game.get_board_of_previous_player(), action_probabilities,
                                                   game.previous_player)
                if game.has_game_ended():
                    finished_games.append(game)
                    training_samples_from_game = training_sample_generator.get_training_data(game.winner)
                    training_samples.extend(training_samples_from_game)
            self.games = [game for game in self.games if not game in finished_games]
            self.print_progress(len(finished_games) / number_of_games)

        return (finished_games, training_samples, self.get_average_game_length(finished_games))

    def get_average_game_length(self, games):
        sum_game_length = 0
        for game in games:
            sum_game_length += game.num_steps
        return sum_game_length / len(games)

    def get_maps_of_games(self):
        maps = []
        for game in self.games:
            maps.append(game.map)
        return maps

    def get_next_agent(self, game):
        if game.previous_player == Player.X:
            return self.o_agent
        else:
            return self.x_agent

    def print_progress(self, progress):
        if self.log_progress:
            barLength = 20
            status = ""
            if progress >= 1:
                progress = 1
                status = "Done...\r\n"
            block = int(round(barLength * progress))
            text = "\r[{0}] {1}% {2}".format("#" * block + "-" * (barLength - block), progress * 100,
                                             status)
            sys.stdout.write(text)
            sys.stdout.flush()
