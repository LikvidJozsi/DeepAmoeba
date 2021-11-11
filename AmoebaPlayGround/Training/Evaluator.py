import collections
import math

from AmoebaPlayGround.Agents.AmoebaAgent import AmoebaAgent, RandomAgent
from AmoebaPlayGround.Agents.HandWrittenAgent import HandWrittenAgent
from AmoebaPlayGround.Training.Puzzles import PuzzleEvaluator

ReferenceAgent = collections.namedtuple('ReferenceAgent', 'name instance evaluation_match_count')
fix_reference_agents = [ReferenceAgent(name='random_agent', instance=RandomAgent(),
                                       evaluation_match_count=24),
                        ReferenceAgent(name='hand_written_agent', instance=HandWrittenAgent(),
                                       evaluation_match_count=24)
                        ]


class Evaluator:
    def evaluate_agent(self, agent: AmoebaAgent, logger):
        pass

    def set_reference_agent(self, agent: AmoebaAgent, rating):
        pass


class EloEvaluator(Evaluator):
    def __init__(self, game_executor, evaluation_match_count=36, puzzle_variation_count=50):
        self.game_executor = game_executor
        self.reference_agent = None
        self.reference_agent_rating = None
        self.evaluation_match_count = evaluation_match_count + evaluation_match_count % 2
        self.puzzle_evaluator = PuzzleEvaluator(puzzle_variation_count)

    def evaluate_agent(self, agent: AmoebaAgent, logger):
        self.evaluate_against_fixed_references(agent, logger)
        if self.reference_agent is not None:
            rating = self.evaluate_against_previous_version(agent_to_evaluate=agent,
                                                            reference_agent=self.reference_agent)
        else:
            rating = 0
        self.reference_agent_rating = rating
        logger.log("agent_rating", rating)
        print('Learning agent rating: %f' % rating)
        self.puzzle_evaluator.evaluate_agent(agent, logger)
        return rating

    def evaluate_against_fixed_references(self, agent_to_evaluate, logger=None):
        for reference_agent in fix_reference_agents:
            games_agent_won, draw_count, avg_game_length = self.play_matches(agent_to_evaluate=agent_to_evaluate,
                                                                             reference_agent=reference_agent.instance,
                                                                             evaluation_match_count=reference_agent.evaluation_match_count)
            score = (games_agent_won + 0.5 * draw_count) / reference_agent.evaluation_match_count
            if logger is not None:
                logger.log(reference_agent.name + "_score", score)
                logger.log(reference_agent.name + "_game_length", avg_game_length)
            print('Score against %s: %f' % (reference_agent.name, score))

    def evaluate_against_previous_version(self, agent_to_evaluate, reference_agent):
        match_count = self.evaluation_match_count
        games_agent_won, draw_count, avg_game_length = self.play_matches(agent_to_evaluate=agent_to_evaluate,
                                                                         reference_agent=reference_agent,
                                                                         evaluation_match_count=match_count)
        if games_agent_won == 0:
            games_agent_won += 1
            match_count += 1
        if games_agent_won == match_count:
            match_count += 1
        agent_expected_score = (games_agent_won + 0.5 * draw_count) / match_count
        print(agent_expected_score)
        agent_rating = self.reference_agent_rating - 400 * math.log10(1 / agent_expected_score - 1)
        return agent_rating

    def play_matches(self, agent_to_evaluate, reference_agent, evaluation_match_count):
        games, _, statistics = self.game_executor.play_games_between_agents(
            evaluation_match_count, agent_to_evaluate, reference_agent, evaluation=True, print_progress=True)
        return statistics.games_won_by_player_1, statistics.draw_games, statistics.get_average_game_length()

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
