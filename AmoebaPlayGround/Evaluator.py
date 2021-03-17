import collections

import math

from AmoebaPlayGround.AmoebaAgent import AmoebaAgent, RandomAgent
from AmoebaPlayGround.GameBoard import Player
from AmoebaPlayGround.GameGroup import GameGroup
from AmoebaPlayGround.HandWrittenAgent import HandWrittenAgent

ReferenceAgent = collections.namedtuple('ReferenceAgent', 'name instance evaluation_match_count')
fix_reference_agents = [ReferenceAgent(name='RandomAgent', instance=RandomAgent(),
                                       evaluation_match_count=100),
                        ReferenceAgent(name='HandWrittenAgent', instance=HandWrittenAgent(),
                                       evaluation_match_count=50)
                        ]

class Evaluator:
    def evaluate_agent(self, agent: AmoebaAgent):
        pass

    def set_reference_agent(self, agent: AmoebaAgent, rating):
        pass


class EloEvaluator(Evaluator):
    def __init__(self, evaluation_match_count=100):
        self.reference_agent = None
        self.reference_agent_rating = None
        self.evaluation_match_count = evaluation_match_count

    def evaluate_agent(self, agent: AmoebaAgent):
        scores_against_fixed = self.evaluate_against_fixed_references(agent)
        if self.reference_agent is not None:
            return scores_against_fixed, self.evaluate_against_agent(agent_to_evaluate=agent,
                                                                 reference_agent=self.reference_agent)
        else:
            return scores_against_fixed, 0
    def evaluate_against_fixed_references(self, agent_to_evaluate):
        scores = {}
        for reference_agent in fix_reference_agents:
            score = self.calculate_expected_score(agent_to_evaluate=agent_to_evaluate,
                                                  reference_agent=reference_agent.instance,
                                                  evaluation_match_count=reference_agent.evaluation_match_count)
            scores[reference_agent.name] = score
            print('Score against %s: %f' % (reference_agent.name, score))
        return scores

    def evaluate_against_agent(self, agent_to_evaluate, reference_agent):
        agent_expected_score = self.calculate_expected_score_for_rating(agent_to_evaluate=agent_to_evaluate,
                                                             reference_agent=reference_agent,
                                                             evaluation_match_count=self.evaluation_match_count)
        agent_rating = self.reference_agent_rating - 400 * math.log10(1 / agent_expected_score - 1)
        return agent_rating

    def calculate_expected_score(self, agent_to_evaluate, reference_agent, evaluation_match_count):
        game_group_size = int(evaluation_match_count / 2)
        game_group_reference_starts = GameGroup(game_group_size,
                                                reference_agent, agent_to_evaluate)
        game_group_agent_started = GameGroup(game_group_size,
                                             agent_to_evaluate, reference_agent)
        finished_games_reference_started, _ = game_group_reference_starts.play_all_games()
        finished_games_agent_started, _ = game_group_agent_started.play_all_games()

        games_agent_won, games_reference_won, draw_games = self.get_win_statistics(finished_games_agent_started)
        won_by_reference, lost_by_reference, draw = self.get_win_statistics(finished_games_reference_started)
        games_agent_won += lost_by_reference
        games_reference_won += won_by_reference
        draw_games += draw
        all_games_num = games_agent_won + games_reference_won + draw_games
        agent_expected_score = games_agent_won / all_games_num + 0.5 * draw_games / all_games_num
        return agent_expected_score

    def calculate_expected_score_for_rating(self, agent_to_evaluate, reference_agent, evaluation_match_count):
        game_group_size = int(evaluation_match_count / 2)
        game_group_reference_starts = GameGroup(game_group_size,
                                                reference_agent, agent_to_evaluate)
        game_group_agent_started = GameGroup(game_group_size,
                                             agent_to_evaluate, reference_agent)
        finished_games_reference_started, _ = game_group_reference_starts.play_all_games()
        finished_games_agent_started, _ = game_group_agent_started.play_all_games()

        games_agent_won, games_reference_won, draw_games = self.get_win_statistics(finished_games_agent_started)
        won_by_reference, lost_by_reference, draw = self.get_win_statistics(finished_games_reference_started)
        games_agent_won += lost_by_reference + 1
        games_reference_won += won_by_reference + 1
        draw_games += draw + 1
        all_games_num = games_agent_won + games_reference_won + draw_games
        agent_expected_score = games_agent_won / all_games_num + 0.5 * draw_games / all_games_num
        return agent_expected_score

    def get_win_statistics(self, games):
        games_x_won = 0
        games_o_won = 0
        games_draw = 0
        for game in games:
            winner = game.winner
            if winner == Player.X:
                games_x_won += 1
            elif winner == Player.O:
                games_o_won += 1
            else:
                games_draw += 1
        return games_x_won, games_o_won, games_draw

    def set_reference_agent(self, agent: AmoebaAgent, rating=1000):
        self.reference_agent = agent
        self.reference_agent_rating = rating

# 1. evaluator recieves an agent
# 2. takes this agent and runs a 1000 games between it and the agent evaluated against ( half one staring half the other)
# 3. calculates elo rating from the elo rating of the reference agent and the win ratio
# 4. returns the win ratio and elo rating
# 5. initially elo of the first agent is 0, what we evaluate against is the previous episode agent
# future ideas:
# what if elo rating is not consistent when evaluating against multiple agents?
