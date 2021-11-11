import glob
import os
from abc import ABC, abstractmethod
from typing import List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import numpy as np
import ray
import tensorflow as tf
from tensorflow.keras.models import Model

import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.Agents.AmoebaAgent import AmoebaAgent
from AmoebaPlayGround.Agents.TensorflowModels import NetworkModel

models_folder = 'Models/'

tf.config.optimizer.set_jit(True)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class NeuralAgent(AmoebaAgent, ABC):
    def __init__(self, model_type: NetworkModel, model_name=None, load_latest_model=False, map_size=(8, 8)):
        self.model_type = model_type
        self.copy_setter_methods: List = []
        self.map_size = map_size
        if load_latest_model:
            latest_model_file = self.get_latest_model()
            print("\n\nLoading model contained in file: %s\n\n" % (latest_model_file))
            self.load_model(latest_model_file)
        else:
            if model_name is None:
                self.model: Model = model_type.create_model(self.map_size)
            else:
                self.load_model(self.get_model_file_path(model_name))

    def get_copy(self):
        new_instance = self.__class__(model_type=self.model_type)
        new_instance.set_weights(self.get_weights())
        return new_instance

    def add_synchronized_copy(self, copy):
        self.copy_setter_methods.append(copy)

    def get_latest_model(self):
        list_of_files = glob.glob(os.path.join(models_folder, '*.h5'))
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file

    def load_model(self, file_path):
        self.model: Model = tf.keras.models.load_model(file_path)

    def get_model_file_path(self, model_name):
        return os.path.join(models_folder, model_name + '.h5')

    def save(self, model_name):
        self.model.save(self.get_model_file_path(model_name))

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)
        self.distribute_weights()

    def distribute_weights(self):
        weights = self.get_weights()
        operations = []
        for copy_method in self.copy_setter_methods:
            operations.append(copy_method.remote(weights))
        for operation in operations:
            ray.get(operation)

    def copy_weights_into(self, agent_to_copy_into):
        agent_to_copy_into.set_weights(self.get_weights())

    def print_model_summary(self):
        self.model.summary()

    @abstractmethod
    def get_name(self):
        return 'NeuralAgent'

    @abstractmethod
    def get_step(self, games: List[Amoeba.AmoebaGame], player, evaluation=False):
        pass

    @abstractmethod
    def train(self, training_samples: List[np.ndarray]):
        self.model_type.train(self.model, training_samples)