from AmoebaPlayGround import AmoebaAgent
from AmoebaPlayGround.Evaluator import EloEvaluator
from AmoebaPlayGround.GameParallelizer import ParallelGameExecutor
from AmoebaPlayGround.Logger import Logger, Statistics
from AmoebaPlayGround.TrainingSampleGenerator import TrainingSampleCollection


class AmoebaTrainer:
    def __init__(self, learning_agent, teaching_agents, self_play=True, trainingset_size=20000, game_executor=None,
                 worker_count=2):
        self.learning_agent: AmoebaAgent = learning_agent
        self.learning_agent_with_old_state: AmoebaAgent = learning_agent.get_copy()
        self.teaching_agents = teaching_agents

        if game_executor is None:
            game_executor = ParallelGameExecutor(learning_agent, self.learning_agent_with_old_state, worker_count)

        self.evaluator = EloEvaluator(game_executor)
        self.game_executor = game_executor
        self.self_play = self_play
        self.training_samples = TrainingSampleCollection(max_size=trainingset_size)
        if self.self_play:
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
                _, training_samples_from_agent, group_statistics = self.game_executor.play_games_between_agents(
                    self.batch_size, self.learning_agent, teaching_agent, evaluation=False)
                print('Average game length against %s: %f' % (
                    teaching_agent.get_name(), group_statistics.get_average_game_length()))
                self.training_samples.extend(training_samples_from_agent)
                statistics.merge_statistics(group_statistics)
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
