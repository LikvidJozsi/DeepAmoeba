import glob
import os
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Add, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.AmoebaAgent import AmoebaAgent
from AmoebaPlayGround.GameBoard import AmoebaBoard, Symbol, Player
from AmoebaPlayGround.RewardCalculator import TrainingSample

models_folder = 'Models/'


class NetworkModel:
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
        self.graph = tf.Graph()
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

    def get_step(self, game_boards: List[AmoebaBoard]):
        output = self.get_model_output(game_boards)
        return self.get_steps_from_output(output, game_boards)

    def get_steps_from_output(self, output, game_boards: List[AmoebaBoard]):
        steps = []
        for probabilities, game_board in zip(output, game_boards):
            valid_steps, valid_probabilities = self.get_valid_steps(probabilities, game_board)

            exploitation_probability = np.random.uniform(0, 1)
            if exploitation_probability < 0.4:
                max_index = np.argmax(valid_probabilities)
                step_in_1d = valid_steps[max_index]
            else:
                step_in_1d = np.random.choice(valid_steps, p=valid_probabilities)

            # step_in_1d = np.random.choice(valid_steps, p=valid_probabilities)

            step = self.to_2d(step_in_1d)
            steps.append(step)
        return steps

    def get_valid_steps(self, probabilites, game_board):
        valid_steps = []
        index = 0
        for row in game_board:
            for cell in row:
                if cell == Symbol.EMPTY:
                    valid_steps.append(index)
                index += 1
        valid_probabilities = probabilites[valid_steps]
        return valid_steps, valid_probabilities / np.sum(valid_probabilities)

    def to_2d(self, index_1d):
        return (int(np.floor(index_1d / self.map_size[0])), index_1d % self.map_size[0])

    def to_1d(self, index_2d):
        return int(index_2d[0] * self.map_size[0] + index_2d[1])

    def get_model_output(self, game_boards: List[AmoebaBoard]):
        formatted_input = self.format_input(game_boards)
        output = self.model.predict(formatted_input, batch_size=256)
        # disclaimer: output has only one spatial dimension, the map is flattened
        return output

    def format_input(self, game_boards: List[AmoebaBoard]):
        numeric_input = []
        for game_board in game_boards:
            numeric_input.append(self.one_hot_encode_input(game_board))
        return np.array(numeric_input)

    def one_hot_encode_input(self, input):
        numeric_representation = np.zeros(input.shape + (2,))
        for row_index, row in enumerate(input.cells):
            for column_index, cell in enumerate(row):
                if input.perspective == Player.X:
                    dimension_for_x = 0
                else:
                    dimension_for_x = 1
                if cell == Symbol.X:
                    numeric_representation[row_index, column_index, dimension_for_x] = 1
                elif cell == Symbol.O:
                    numeric_representation[row_index, column_index, 1 - dimension_for_x] = 1
        return numeric_representation

    def one_hot_encode_output(self, output):
        expanded_dims = np.zeros(np.prod(self.map_size))
        output_1d = self.to_1d(output)
        expanded_dims[output_1d] = 1
        return expanded_dims

    def one_hot_encode_outputs(self, outputs):
        one_hot_outputs = list(map(self.one_hot_encode_output, outputs))
        return np.array(one_hot_outputs)

    def train(self, training_samples: List[TrainingSample]):
        input, output, weights = TrainingSample.unpack(training_samples)
        output = self.one_hot_encode_outputs(output)
        print('number of training samples: ' + str(len(training_samples)))
        input = np.array([self.one_hot_encode_input(x) for x in input])
        return self.model.fit(x=input, y=np.array(output), sample_weight=np.array(weights), epochs=15,
                              shuffle=True,
                              verbose=2, batch_size=32)

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
