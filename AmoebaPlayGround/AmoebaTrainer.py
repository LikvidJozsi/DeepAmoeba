from AmoebaPlayGround import AmoebaAgent
from AmoebaPlayGround.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.Evaluator import EloEvaluator
from AmoebaPlayGround.GameGroup import GameGroup
from AmoebaPlayGround.Logger import Logger, Statistics
from AmoebaPlayGround.MoveSelector import DistributionMoveSelector, MaximalMoveSelector
from AmoebaPlayGround.TrainingSampleGenerator import TrainingSampleCollection, \
    SymmetricTrainingSampleGenerator


class AmoebaTrainer:
    def __init__(self, learning_agent, teaching_agents,
                 training_sample_generator_class=SymmetricTrainingSampleGenerator, self_play=True,
                 training_move_selector=DistributionMoveSelector(), evaluation_move_selector=MaximalMoveSelector(),
                 trainingset_size=20000):
        self.learning_agent: AmoebaAgent = learning_agent
        self.training_sample_generator_class = training_sample_generator_class
        self.teaching_agents = teaching_agents
        self.training_move_selector = training_move_selector
        self.evaluation_move_selector = evaluation_move_selector
        self.evaluator = EloEvaluator(move_selector=evaluation_move_selector,
                                      self_play_move_selector=training_move_selector)
        self.self_play = self_play
        self.training_samples = TrainingSampleCollection(max_size=trainingset_size)
        if self.self_play:
            self.learning_agent_with_old_state = BatchMCTSAgent(model_type=self.learning_agent.model_type)
            self.teaching_agents.append(self.learning_agent)

    def train(self, batch_size=1, view=None, num_episodes=1, model_save_file="", logger=Logger()):
        self.batch_size = batch_size
        self.view = view

        if self.self_play:
            self.evaluator.set_reference_agent(self.learning_agent_with_old_state)
        for episode_index in range(num_episodes):
            logger.log("episode", episode_index)
            print('\nEpisode %d:' % episode_index)
            statistics = Statistics()
            for teacher_index, teaching_agent in enumerate(self.teaching_agents):
                print('Playing games against ' + teaching_agent.get_name())
                training_samples_from_agent, group_statistics = self.play_games_between_agents(self.learning_agent,
                                                                                               teaching_agent)
                print('Average game length against %s: %f' % (
                    teaching_agent.get_name(), group_statistics.get_average_game_length()))
                self.training_samples.extend(training_samples_from_agent)
                statistics.merge_statistics(group_statistics)
            if self.self_play:
                self.learning_agent.copy_weights_into(self.learning_agent_with_old_state)
            statistics.log(logger)
            print('Training agent:')
            train_history = self.learning_agent.train(self.training_samples)
            last_loss = train_history.history['loss'][-1]
            logger.log("loss", last_loss)

            print('Evaluating agent:')
            agent_rating = self.evaluator.evaluate_agent(self.learning_agent, logger)

            print('Learning agent rating: %f' % agent_rating)
            logger.new_episode()
            if model_save_file != "":
                self.learning_agent.save(model_save_file)

    def play_games_between_agents(self, agent_one, agent_two):
        game_group_1 = GameGroup(max(1, int(self.batch_size / 2)), agent_one, agent_two,
                                 self.view, log_progress=True,
                                 training_sample_generator_class=self.training_sample_generator_class,
                                 move_selector=self.training_move_selector)
        game_group_2 = GameGroup(max(1, int(self.batch_size / 2)), agent_two, agent_one,
                                 self.view, log_progress=True,
                                 training_sample_generator_class=self.training_sample_generator_class,
                                 move_selector=self.training_move_selector)

        _, training_samples_1, statistics_1 = game_group_1.play_all_games()
        _, training_samples_2, statistics_2 = game_group_2.play_all_games()
        training_samples_1.extend(training_samples_2)
        statistics_1.merge_statistics(statistics_2)
        return training_samples_1, statistics_1
