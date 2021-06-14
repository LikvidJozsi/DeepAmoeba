import collections
import math

from AmoebaPlayGround.AmoebaAgent import AmoebaAgent, RandomAgent
from AmoebaPlayGround.GameBoard import Player
from AmoebaPlayGround.GameGroup import GameGroup
from AmoebaPlayGround.HandWrittenAgent import HandWrittenAgent
from AmoebaPlayGround.MoveSelector import MaximalMoveSelector, DistributionMoveSelector

ReferenceAgent = collections.namedtuple('ReferenceAgent', 'name instance evaluation_match_count')
fix_reference_agents = [ReferenceAgent(name='random_agent', instance=RandomAgent(),
                                       evaluation_match_count=10),
                        ReferenceAgent(name='hand_written_agent', instance=HandWrittenAgent(),
                                       evaluation_match_count=10)
                        ]


class Evaluator:
    def evaluate_agent(self, agent: AmoebaAgent):
        pass

    def set_reference_agent(self, agent: AmoebaAgent, rating):
        pass


class EloEvaluator(Evaluator):
    def __init__(self, evaluation_match_count=20, move_selector=MaximalMoveSelector(),
                 self_play_move_selector=DistributionMoveSelector()):
        self.reference_agent = None
        self.reference_agent_rating = None
        self.move_selector = move_selector
        self.self_play_move_selector = self_play_move_selector
        self.evaluation_match_count = evaluation_match_count + evaluation_match_count % 2

    def evaluate_agent(self, agent: AmoebaAgent):
        scores_against_fixed, length_against_fixed = self.evaluate_against_fixed_references(agent)
        if self.reference_agent is not None:
            return scores_against_fixed, length_against_fixed, self.evaluate_against_previous_version(
                agent_to_evaluate=agent,
                reference_agent=self.reference_agent)
        else:
            return scores_against_fixed, length_against_fixed, 0

    def evaluate_against_fixed_references(self, agent_to_evaluate):
        scores = {}
        avg_game_lengths = {}
        for reference_agent in fix_reference_agents:
            games_agent_won, draw_count, avg_game_length = self.play_matches(agent_to_evaluate=agent_to_evaluate,
                                                                             reference_agent=reference_agent.instance,
                                                                             evaluation_match_count=reference_agent.evaluation_match_count)
            score = (games_agent_won + 0.5 * draw_count) / reference_agent.evaluation_match_count
            scores[reference_agent.name] = score
            avg_game_lengths[reference_agent.name] = avg_game_length
            print('Score against %s: %f' % (reference_agent.name, score))
        return scores, avg_game_lengths

    def evaluate_against_previous_version(self, agent_to_evaluate, reference_agent):
        match_count = self.evaluation_match_count
        games_agent_won, draw_count, avg_game_length = self.play_matches(
            agent_to_evaluate=agent_to_evaluate,
            reference_agent=reference_agent,
            evaluation_match_count=match_count)
        if games_agent_won == 0:
            games_agent_won += 1
            match_count += 1
        if games_agent_won == match_count:
            match_count += 1
        agent_expected_score = (games_agent_won + 0.5 * draw_count) / match_count
        agent_rating = self.reference_agent_rating - 400 * math.log10(1 / agent_expected_score - 1)
        return agent_rating

    def play_matches(self, agent_to_evaluate, reference_agent, evaluation_match_count):
        game_group_size = math.ceil(evaluation_match_count / 2)
        game_group_reference_starts = GameGroup(game_group_size,
                                                reference_agent, agent_to_evaluate, log_progress=True,
                                                move_selector=self.move_selector)
        game_group_agent_started = GameGroup(game_group_size,
                                             agent_to_evaluate, reference_agent, log_progress=True,
                                             move_selector=self.move_selector)
        finished_games_reference_started, _, avg_game_length_1 = game_group_reference_starts.play_all_games()
        finished_games_agent_started, _, avg_game_length_2 = game_group_agent_started.play_all_games()
        aggregate_avg_game_length = (avg_game_length_1 + avg_game_length_2) / 2
        agent_win_1, reference_win_1, draw_1 = self.get_win_statistics(finished_games_agent_started)
        reference_win_2, agent_win_2, draw_2 = self.get_win_statistics(finished_games_reference_started)
        aggregate_agent_win = agent_win_1 + agent_win_2
        aggregate_reference_win = reference_win_1 + reference_win_2
        draw_count = draw_1 + draw_2
        return aggregate_agent_win, draw_count, aggregate_avg_game_length

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
