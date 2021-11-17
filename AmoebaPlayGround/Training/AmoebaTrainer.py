import glob
import os
import pickle

from AmoebaPlayGround.Agents import AmoebaAgent
from AmoebaPlayGround.GameExecution.GameParallelizer import ParallelGameExecutor
from AmoebaPlayGround.Training.Evaluator import EloEvaluator
from AmoebaPlayGround.Training.Input import get_model_filename
from AmoebaPlayGround.Training.Logger import Statistics, FileLogger
from AmoebaPlayGround.Training.Logger import logs_folder
from AmoebaPlayGround.Training.TrainingSampleGenerator import TrainingSampleCollection


class AmoebaTrainer:
    def __init__(self, learning_agent, teaching_agents, self_play=True, trainingset_size=20000, game_executor=None,
                 worker_count=2, training_sample_entropy_cutoff_schedule=None, resume_previous_training=False,
                 training_sample_turn_cutoff_schedule=None):
        self.learning_agent: AmoebaAgent = learning_agent
        self.learning_agent_with_old_state: AmoebaAgent = learning_agent.get_copy()
        self.teaching_agents = teaching_agents
        self.training_sample_entropy_cutoff_schedule = training_sample_entropy_cutoff_schedule
        self.training_sample_turn_cutoff_schedule = training_sample_turn_cutoff_schedule

        if game_executor is None:
            game_executor = ParallelGameExecutor(learning_agent, self.learning_agent_with_old_state, worker_count)

        self.evaluator = EloEvaluator(game_executor)
        self.game_executor = game_executor
        self.self_play = self_play

        if resume_previous_training:
            self.training_id = self.get_latest_training_id()
            self.training_samples = self.load_latest_dataset()
            self.logger = FileLogger(self.training_id)
            self.current_episode = self.logger.get_log_episode_count()
        else:
            self.training_samples = TrainingSampleCollection(max_size=trainingset_size)
            self.training_id = get_model_filename()
            self.logger = FileLogger(self.training_id)
            self.current_episode = 0

        if self.self_play:
            self.teaching_agents.append(self.learning_agent)

    def get_latest_training_id(self):
        list_of_files = glob.glob(os.path.join(logs_folder, '*.csv'))
        latest_file_path = max(list_of_files, key=os.path.getctime)
        latest_file_name_with_extension = os.path.basename(latest_file_path)
        latest_file_name, extension = os.path.splitext(latest_file_name_with_extension)
        return latest_file_name

    def get_scheduled_value_for_episode(self, schedule, episode):
        if schedule is None:
            return None
        for index, (episode_start, value) in enumerate(schedule):
            if episode_start > episode:
                return schedule[index - 1][1]
        return schedule[-1][1]

    def recalculate_scheduled_parameters(self, episode_index):
        training_sample_turn_cutoff = self.get_scheduled_value_for_episode(self.training_sample_turn_cutoff_schedule,
                                                                           episode_index)
        training_sample_entropy_cutoff = self.get_scheduled_value_for_episode(
            self.training_sample_entropy_cutoff_schedule,
            episode_index)
        return training_sample_turn_cutoff, training_sample_entropy_cutoff

    def load_latest_dataset(self):
        with open("Datasets/latest_dataset.p", "rb") as file:
            dataset = pickle.load(file)
            return dataset

    def save_latest_dataset(self, dataset):
        with open("Datasets/latest_dataset.p", "wb") as file:
            pickle.dump(dataset, file)

    def train(self, batch_size=1, batches_per_episode=1, view=None, num_episodes=1):
        self.batch_size = batch_size
        self.view = view

        if self.self_play:
            self.evaluator.set_reference_agent(self.learning_agent_with_old_state)
        while self.current_episode < num_episodes:
            self.logger.log("episode", self.current_episode)
            training_sample_turn_cutoff, training_sample_entropy_cutoff = self.recalculate_scheduled_parameters(
                self.current_episode)
            statistics = Statistics()
            print(f'\nEpisode {self.current_episode}, training_sample_turn_cutoff:{training_sample_turn_cutoff}, '
                  f'training_sample_entropy_cutoff: {training_sample_entropy_cutoff}')
            for teacher_index, teaching_agent in enumerate(self.teaching_agents):
                # print('Playing games against ' + teaching_agent.get_name())
                for game_batch_index in range(batches_per_episode):
                    _, training_samples_from_agent, group_statistics = self.game_executor.play_games_between_agents(
                        self.batch_size, self.learning_agent, teaching_agent, evaluation=False, print_progress=True)

                    training_samples_from_agent.create_rotational_variations(training_sample_entropy_cutoff,
                                                                             training_sample_turn_cutoff)
                    self.training_samples.extend(training_samples_from_agent)
                    statistics.merge_statistics(group_statistics)
                print('Average game length against %s: %f' % (
                    teaching_agent.get_name(), statistics.get_average_game_length()))
            self.learning_agent.copy_weights_into(self.learning_agent_with_old_state)
            statistics.log(self.logger)
            print('Training agent:')
            train_history = self.learning_agent.train(self.training_samples)
            self.learning_agent.distribute_weights()
            last_loss = train_history.history['loss'][-1]
            self.logger.log("loss", last_loss)

            print('Evaluating agent:')
            self.evaluator.evaluate_agent(self.learning_agent, self.logger)

            self.logger.new_episode()
            self.learning_agent.save(self.training_id)
            self.save_latest_dataset(self.training_samples)
            self.current_episode += 1
