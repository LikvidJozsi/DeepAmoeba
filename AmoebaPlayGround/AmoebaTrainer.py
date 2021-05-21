from AmoebaPlayGround import AmoebaAgent
from AmoebaPlayGround.Evaluator import EloEvaluator
from AmoebaPlayGround.Evaluator import fix_reference_agents
from AmoebaPlayGround.GameGroup import GameGroup
from AmoebaPlayGround.Logger import Logger
from AmoebaPlayGround.MCTSAgent import MCTSAgent
from AmoebaPlayGround.MoveSelector import DistributionMoveSelector, MaximalMoveSelector
from AmoebaPlayGround.TrainingSampleGenerator import TrainingSampleCollection, \
    SymmetricTrainingSampleGenerator


class AmoebaTrainer:
    def __init__(self, learning_agent, teaching_agents,
                 training_sample_generator_class=SymmetricTrainingSampleGenerator, self_play=True,
                 training_move_selector=DistributionMoveSelector(), evaluation_move_selector=MaximalMoveSelector()):
        self.learning_agent: AmoebaAgent = learning_agent
        self.training_sample_generator_class = training_sample_generator_class
        self.teaching_agents = teaching_agents
        self.training_move_selector = training_move_selector
        self.evaluation_move_selector = evaluation_move_selector
        self.evaluator = EloEvaluator(move_selector=evaluation_move_selector,
                                      self_play_move_selector=training_move_selector)
        self.self_play = self_play
        if self.self_play:
            self.learning_agent_with_old_state = MCTSAgent(model_type=self.learning_agent.model_type)
            self.teaching_agents.append(self.learning_agent)

    def train(self, batch_size=1, view=None, num_episodes=1, model_save_file="", logger=Logger()):
        self.batch_size = batch_size
        self.view = view

        if self.self_play:
            self.evaluator.set_reference_agent(self.learning_agent_with_old_state)
        for episode_index in range(num_episodes):
            logger.log_value(episode_index)
            print('\nEpisode %d:' % episode_index)
            aggregate_average_game_length = 0
            training_samples = TrainingSampleCollection()
            for teacher_index, teaching_agent in enumerate(self.teaching_agents):
                print('Playing games against ' + teaching_agent.get_name())
                training_samples_from_agent, average_game_length = self.play_games_between_agents(self.learning_agent,
                                                                                                  teaching_agent)
                print('Average game length against %s: %f' % (teaching_agent.get_name(), average_game_length))
                training_samples.extend(training_samples_from_agent)
                aggregate_average_game_length += average_game_length
            aggregate_average_game_length /= float(len(self.teaching_agents))
            logger.log_value(aggregate_average_game_length)
            self.learning_agent.reset()

            if self.self_play:
                self.learning_agent.copy_weights_into(self.learning_agent_with_old_state)
            print('Training agent:')
            train_history = self.learning_agent.train(training_samples)
            last_loss = train_history.history['loss'][-1]
            logger.log_value(last_loss)

            print('Evaluating agent:')
            scores_against_fixed, agent_rating = self.evaluator.evaluate_agent(self.learning_agent)
            logger.log_value(agent_rating)
            for reference_agent in fix_reference_agents:
                logger.log_value(scores_against_fixed[reference_agent.name])
            print('Learning agent rating: %f' % agent_rating)
            logger.newline()
            if model_save_file != "":
                self.learning_agent.save(model_save_file)
        logger.close()

    def play_games_between_agents(self, agent_one, agent_two):
        game_group_1 = GameGroup(max(1, int(self.batch_size / 2)), agent_one, agent_two,
                                 self.view, log_progress=True,
                                 training_sample_generator_class=self.training_sample_generator_class,
                                 move_selector=self.training_move_selector)
        game_group_2 = GameGroup(max(1, int(self.batch_size / 2)), agent_two, agent_one,
                                 self.view, log_progress=True,
                                 training_sample_generator_class=self.training_sample_generator_class,
                                 move_selector=self.training_move_selector)

        _, training_samples_1, average_game_length_1 = game_group_1.play_all_games()
        _, training_samples_2, average_game_length_2 = game_group_2.play_all_games()
        training_samples_1.extend(training_samples_2)
        return training_samples_1, (average_game_length_1 + average_game_length_2) / 2
