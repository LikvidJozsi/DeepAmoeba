import glob
import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.AmoebaAgent import AmoebaAgent
from AmoebaPlayGround.NetworkModels import NetworkModel

models_folder = 'Models/'


class NeuralAgent(AmoebaAgent, ABC):
    def __init__(self, model_type: NetworkModel, model_name=None, load_latest_model=False):
        self.model_type = model_type
        if load_latest_model:
            latest_model_file = self.get_latest_model()
            print("\n\nLoading model contained in file: %s\n\n" % (latest_model_file))
            self.load_model(latest_model_file)
        else:
            if model_name is None:
                self.map_size = Amoeba.map_size
                self.model: Model = model_type.create_model(self.map_size)
            else:
                self.load_model(self.get_model_file_path(model_name))

    def get_latest_model(self):
        list_of_files = glob.glob(os.path.join(models_folder, '*.h5'))
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file

    def load_model(self, file_path):
        self.model: Model = tf.keras.models.load_model(file_path)
        self.map_size = self.model.get_layer(index=0).output_shape[1:3]

    def get_model_file_path(self, model_name):
        return os.path.join(models_folder, model_name + '.h5')

    def save(self, model_name):
        self.model.save(self.get_model_file_path(model_name))

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

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
        pass
