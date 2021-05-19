import glob
import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Add, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.AmoebaAgent import AmoebaAgent
from AmoebaPlayGround.GameBoard import AmoebaBoard
from AmoebaPlayGround.TrainingSampleGenerator import TrainingSample

models_folder = 'Models/'


class NetworkModel(ABC):
    @abstractmethod
    def create_model(self, map_size):
        pass


class ShallowNetwork(NetworkModel):
    def __init__(self, first_convolution_size=(9, 9)):
        self.first_convolution_size = first_convolution_size

    def create_model(self, map_size):
        input = Input(shape=map_size + (2,))
        conv_1 = Conv2D(32, kernel_size=self.first_convolution_size, activation='relu', padding='same')(input)
        conv_2 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(conv_1)
        pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_2)
        conv_3 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(pooling)
        flatten = Flatten()(conv_3)
        dense_1 = Dense(256, activation='relu')(flatten)
        output = Dense(np.prod(map_size), activation='softmax')(dense_1)
        model = Model(inputs=input, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.3))
        return model


class ResNetLike(NetworkModel):
    def __init__(self, network_depth=8):
        self.network_depth = network_depth

    def create_model(self, map_size):
        input = Input(shape=map_size + (2,))
        conv_1 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        activity_regularizer=regularizers.l2(0.0001))(input)
        batch_norm_1 = BatchNormalization()(conv_1)
        relu_1 = Activation('relu')(batch_norm_1)
        current_network_end = relu_1
        for index in range(self.network_depth):
            current_network_end = self.identity_block(current_network_end, filters=[128, 128])
        dimension_reducer_convolution = Conv2D(2, kernel_size=(1, 1), strides=(1, 1))(current_network_end)
        batch_normalization = BatchNormalization()(dimension_reducer_convolution)
        relu = Activation('relu')(batch_normalization)
        flatten = Flatten()(relu)
        output = Dense(np.prod(map_size), activation='softmax')(flatten)
        model = Model(inputs=input, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01))
        return model

    def identity_block(self, input, filters):
        conv_1 = Conv2D(filters[0], kernel_size=(3, 3), strides=(1, 1), padding='same',
                        activity_regularizer=regularizers.l2(0.0001))(input)
        batch_norm_1 = BatchNormalization()(conv_1)
        relu_1 = Activation('relu')(batch_norm_1)
        conv_2 = Conv2D(filters[1], kernel_size=(3, 3), strides=(1, 1), padding='same',
                        activity_regularizer=regularizers.l2(0.0001))(relu_1)
        batch_norm_2 = BatchNormalization()(conv_2)
        skip_connection = Add()([batch_norm_2, input])
        relu_2 = Activation('relu')(skip_connection)
        return relu_2


class NeuralAgent(AmoebaAgent):
    def __init__(self, model_name=None, load_latest_model=False,
                 model_type: NetworkModel = ShallowNetwork()):
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

    def get_step(self, game_boards: List[AmoebaBoard], player):
        pass

    def to_2d(self, index_1d):
        return (int(np.floor(index_1d / self.map_size[0])), index_1d % self.map_size[0])

    def to_1d(self, index_2d):
        return int(index_2d[0] * self.map_size[0] + index_2d[1])

    def train(self, training_samples: List[TrainingSample]):
        pass

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def copy_weights_into(self, agent_to_copy_into):
        agent_to_copy_into.set_weights(self.get_weights())

    def get_name(self):
        return 'NeuralAgent'

    def print_model_saummary(self):
        self.model.summary()
