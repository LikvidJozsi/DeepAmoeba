import glob
import os
import pickle

from AmoebaPlayGround.Agents import AmoebaAgent
from AmoebaPlayGround.GameExecution.MoveSelector import MoveSelectionStrategy
from AmoebaPlayGround.GameExecution.Multithreading.GameParallelizer import ParallelGameExecutor
from AmoebaPlayGround.Training.Evaluator import EloEvaluator
from AmoebaPlayGround.Training.Input import get_model_filename
from AmoebaPlayGround.Training.Logger import Statistics, FileLogger
from AmoebaPlayGround.Training.Logger import logs_folder
from AmoebaPlayGround.Training.TrainingSampleGenerator import TrainingDatasetGenerator


class AmoebaTrainer:
    def __init__(self, learning_agent, teaching_agents, map_size, game_executor=None,
                 workers_per_inference_server=2, training_sample_entropy_cutoff_schedule=None,
                 resume_previous_training=False,
                 training_sample_turn_cutoff_schedule=None, sample_episode_window_width_schedule=None,
                 move_selection_strategy=MoveSelectionStrategy()):
        self.learning_agent: AmoebaAgent = learning_agent
        self.learning_agent_with_old_state: AmoebaAgent = learning_agent.get_copy()
        self.teaching_agents = teaching_agents
        self.training_sample_entropy_cutoff_schedule = training_sample_entropy_cutoff_schedule
        self.training_sample_turn_cutoff_schedule = training_sample_turn_cutoff_schedule
        self.sample_episode_window_width_schedule = sample_episode_window_width_schedule
        self.map_size = map_size

        if game_executor is None:
            game_executor = ParallelGameExecutor(learning_agent, self.learning_agent_with_old_state,
                                                 workers_per_inference_server,
                                                 move_selection_strategy=move_selection_strategy)
        self.game_executor = game_executor

        self.evaluator = EloEvaluator(game_executor, map_size)

        if resume_previous_training:
            self.training_id = self.get_latest_training_id()
            self.training_dataset_generator = self.load_latest_dataset()
            self.logger = FileLogger(self.training_id)
            self.current_episode = self.logger.get_log_episode_count()
            self.evaluator.set_reference_agent(self.learning_agent_with_old_state,
                                               self.logger.get_latest_agent_rating())
        else:
            self.training_dataset_generator = TrainingDatasetGenerator()
            self.training_id = get_model_filename()
            self.logger = FileLogger(self.training_id)
            self.current_episode = 0
            self.evaluator.set_reference_agent(self.learning_agent_with_old_state, 1000)

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

    def calculate_episode_window_width(self):
        sample_episode_window_width = self.get_scheduled_value_for_episode(
            self.sample_episode_window_width_schedule,
            self.current_episode)
        return sample_episode_window_width

    def calculate_training_sample_entropy_cutoff(self):
        return self.get_scheduled_value_for_episode(
            self.training_sample_entropy_cutoff_schedule,
            self.current_episode)

    def calculate_training_sample_cutoff(self):
        return self.get_scheduled_value_for_episode(self.training_sample_turn_cutoff_schedule,
                                                    self.current_episode)

    def load_latest_dataset(self):
        with open("Datasets/latest_dataset.p", "rb") as file:
            dataset = pickle.load(file)
            return dataset

    def save_latest_dataset(self, dataset):
        with open("Datasets/latest_dataset.p", "wb") as file:
            pickle.dump(dataset, file)

    def load_quickstart_dataset(self, training_sample_entropy_cutoff, training_sample_turn_cutoff):
        with open("Datasets/quickstart_dataset.p", "rb") as file:
            dataset = pickle.load(file)
        with open("Datasets/quickstart_dataset_statistics.p", "rb") as file:
            statistics = pickle.load(file)

        dataset.filter_samples(training_sample_entropy_cutoff,
                               training_sample_turn_cutoff)

        return dataset, statistics

    def train(self, batch_size=1, num_episodes=1):
        while self.current_episode < num_episodes:
            self.logger.log("episode", self.current_episode)
            statistics = Statistics()
            training_samples_for_episode = []
            self.training_dataset_generator.set_episode_window_width(self.calculate_episode_window_width())
            self.print_episode_information()

            for teacher_index, teaching_agent in enumerate(self.teaching_agents):
                print('Playing games against ' + teaching_agent.get_name())
                _, training_samples_from_agent, group_statistics = self.game_executor.play_games_between_agents(
                    batch_size, self.learning_agent, teaching_agent, self.map_size, evaluation=False,
                    print_progress=True)

                training_samples_from_agent.filter_samples(self.calculate_training_sample_entropy_cutoff(),
                                                           self.calculate_training_sample_cutoff())

                statistics.merge_statistics(group_statistics)
                training_samples_for_episode.extend(training_samples_from_agent)

            self.training_dataset_generator.add_episode(training_samples_for_episode)
            self.learning_agent.get_neural_network_model().copy_weights_into(
                self.learning_agent_with_old_state.get_neural_network_model())
            statistics.log(self.logger)
            print('Training agent:')
            self.train_learing_agent()

            print('Evaluating agent:')
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

    def train_learing_agent(self):
        train_history = self.learning_agent.train(self.training_dataset_generator)
        self.learning_agent.get_neural_network_model().distribute_weights()
        last_loss = train_history.history['loss'][-1]
        self.logger.log("loss", last_loss)
