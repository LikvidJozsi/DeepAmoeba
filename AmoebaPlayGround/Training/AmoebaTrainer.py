import glob
import os
import pickle

import toml

from AmoebaPlayGround.GameExecution.SingleThreadGameExecutor import SingleThreadGameExecutor
from AmoebaPlayGround.Agents import AmoebaAgent
from AmoebaPlayGround.GameExecution.Multithreading.GameParallelizer import ParallelGameExecutor
from AmoebaPlayGround.Training.Evaluator import EloEvaluator
from AmoebaPlayGround.Training.Input import get_model_filename
from AmoebaPlayGround.Training.Logger import Statistics, FileLogger
from AmoebaPlayGround.Training.Logger import logs_folder
from AmoebaPlayGround.Training.TrainingSampleGenerator import TrainingDatasetGenerator, TrainingSampleCollection

CONFIGS_FOLDER = '../Configs/'

GAME_EXECUTORS = {
    "ParallelGameExecutor": ParallelGameExecutor,
    "SingleThreadGameExecutor": SingleThreadGameExecutor
}


class AmoebaTrainer:
    def __init__(self, learning_agent, teaching_agents, config):
        self.learning_agent: AmoebaAgent = learning_agent
        learning_agent.name = "learning_agent"
        self.config = config["trainer"]
        self.map_size = config["map_size"]  # TODO make this more elegant somehow
        self.learning_agent_with_old_state: AmoebaAgent = learning_agent.get_copy()
        self.learning_agent_with_old_state.name = "old_state_agent"
        self.teaching_agents = teaching_agents

        game_executor_config = self.config["game_executor"]
        game_executor_class = GAME_EXECUTORS[game_executor_config["game_executor_type"]]
        self.game_executor = game_executor_class(learning_agent, self.learning_agent_with_old_state,
                                                 game_executor_config)

        self.evaluator = EloEvaluator(self.game_executor, self.map_size)  # TODO pass config

        if self.config["resume_previous_training"]:
            self.training_id = self.get_latest_training_id()
            self.training_dataset_generator = self.load_latest_dataset()
            self.logger = FileLogger(self.training_id)
            self.current_episode = self.logger.get_log_episode_count()
            self.evaluator.set_reference_agent(self.learning_agent_with_old_state,
                                               self.logger.get_latest_agent_rating())
            self.save_config(self.training_id, config)
        else:
            self.training_dataset_generator = TrainingDatasetGenerator()
            self.training_id = get_model_filename()
            self.logger = FileLogger(self.training_id)
            self.current_episode = 0
            self.evaluator.set_reference_agent(self.learning_agent_with_old_state, 1000)
            self.save_config(self.training_id, config)

    def get_latest_training_id(self):
        list_of_files = glob.glob(os.path.join(logs_folder, '*.csv'))
        latest_file_path = max(list_of_files, key=os.path.getctime)
        latest_file_name_with_extension = os.path.basename(latest_file_path)
        latest_file_name, extension = os.path.splitext(latest_file_name_with_extension)
        return latest_file_name

    def save_config(self, training_id, config):
        with open(self.get_config_file_path(training_id), "w") as config_file:
            toml.dump(config, config_file)

    def get_config_file_path(self, training_id):
        configs_of_training = glob.glob(os.path.join(CONFIGS_FOLDER, training_id + '*.toml'))
        return os.path.join(CONFIGS_FOLDER, training_id + '_' + str(len(configs_of_training)) + '.toml')

    def get_scheduled_value_for_episode(self, schedule, episode):
        if schedule is None:
            return None
        for index, (episode_start, value) in enumerate(schedule):
            if episode_start > episode:
                return schedule[index - 1][1]
        return schedule[-1][1]

    def calculate_episode_window_width(self):
        sample_episode_window_width = self.get_scheduled_value_for_episode(
            self.config.get("sample_episode_window_width_schedule"),
            self.current_episode)
        return sample_episode_window_width

    def calculate_training_sample_entropy_cutoff(self):
        return self.get_scheduled_value_for_episode(
            self.config.get("training_sample_entropy_cutoff_schedule"),
            self.current_episode)

    def calculate_training_sample_cutoff(self):
        return self.get_scheduled_value_for_episode(self.config.get("training_sample_turn_cutoff_schedule"),
                                                    self.current_episode)

    def load_latest_dataset(self):
        with open("../Datasets/latest_dataset.p", "rb") as file:
            dataset = pickle.load(file)
            return dataset

    def save_latest_dataset(self, dataset):
        with open("../Datasets/latest_dataset.p", "wb") as file:
            pickle.dump(dataset, file)

    def train(self):
        while self.current_episode < self.config["episode_count"]:
            self.logger.log("episode", self.current_episode)
            statistics = Statistics()
            training_samples_for_episode = TrainingSampleCollection()
            self.training_dataset_generator.set_episode_window_width(self.calculate_episode_window_width())
            self.print_episode_information()

            for teacher_index, teaching_agent in enumerate(self.teaching_agents):
                print('Playing games against ' + teaching_agent.get_name())
                teaching_agent.set_training_mode()
                self.learning_agent.set_training_mode()
                _, training_samples_from_agent, group_statistics = self.game_executor.play_games_between_agents(
                    self.config["games_per_episode"], self.learning_agent, teaching_agent, self.map_size,
                    evaluation=False,
                    print_progress_active=True)

                training_samples_from_agent.filter_samples(self.calculate_training_sample_entropy_cutoff(),
                                                           self.calculate_training_sample_cutoff())

                statistics.merge_statistics(group_statistics)
                training_samples_for_episode.extend(training_samples_from_agent)

            self.training_dataset_generator.add_episode(training_samples_for_episode)
            self.learning_agent.get_neural_network_model().copy_weights_into(
                self.learning_agent_with_old_state.get_neural_network_model())
            statistics.log(self.logger)
            print('Training agent:')
            self.train_learing_agent(statistics)

            print('Evaluating agent:')
            self.learning_agent.set_evaluation_mode()
            self.learning_agent_with_old_state.set_evaluation_mode()
            self.evaluator.evaluate_agent(self.learning_agent, self.logger)

            self.end_episode()

    def print_episode_information(self):
        print(
            f'\nEpisode {self.current_episode}, training_sample_turn_cutoff:{self.calculate_training_sample_cutoff()}, '
            f'sample_episode_window_width: {self.calculate_episode_window_width()}, sample_episode_count: {len(self.training_dataset_generator.sample_collections)}')

    def end_episode(self):
        self.logger.new_episode()
        self.learning_agent.save(self.training_id)
        self.save_latest_dataset(self.training_dataset_generator)
        self.current_episode += 1

    def train_learing_agent(self, statistics):
        train_history = self.learning_agent.model.train(self.training_dataset_generator,
                                                        fraction_won_by_player_1=statistics.get_fraction_won_by_player_one(),
                                                        fraction_draw=statistics.get_fraction_draw())
        self.logger.log("loss", train_history.history['loss'][-1])
        self.logger.log("policy_loss", train_history.history['policy_loss'][-1])
        self.logger.log("value_loss", train_history.history['value_loss'][-1])
