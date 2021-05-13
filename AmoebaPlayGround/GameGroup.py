import sys

from AmoebaPlayGround.Amoeba import AmoebaGame, Player


class GameGroup:
    def __init__(self, batch_size, x_agent, o_agent, view=None, log_progress=False):
        self.x_agent = x_agent
        self.o_agent = o_agent
        self.log_progress = log_progress
        self.games = []
        for index in range(batch_size):
            self.games.append(AmoebaGame(view))

    def play_all_games(self):
        finished_games = []
        number_of_games = len(self.games)
        while len(self.games) != 0:
            next_agent = self.get_next_agent(self.games[0])  # the same agent has its turn in every active game at the
            # same time, therfore getting the agent of any of them is enough
            maps = self.get_maps_of_games()
            actions = next_agent.get_step(maps, next_agent)
            for index, game in enumerate(self.games):
                game.step(actions[index])
                if game.has_game_ended():
                    finished_games.append(game)
            self.games = [game for game in self.games if not game in finished_games]
            self.print_progress(len(finished_games) / number_of_games)

        return (finished_games, self.get_average_game_length(finished_games))

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
