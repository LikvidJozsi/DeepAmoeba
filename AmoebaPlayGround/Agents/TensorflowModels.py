import glob
import os
from abc import ABC, abstractmethod
from typing import List

import ray
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, BatchNormalization, Activation, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dropout, LeakyReLU
from tensorflow.python.keras.regularizers import l2

models_folder = 'Models/'

tf.config.optimizer.set_jit(True)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_latest_model():
    list_of_files = glob.glob(os.path.join(models_folder, '*.h5'))
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def get_model_file_path(model_name):
    return os.path.join(models_folder, model_name + '.h5')


class NeuralNetworkSkeleton:

    def __init__(self, network_class, config, weights):
        self.network_class = network_class
        self.config = config
        self.weights = weights

    def resurrect_neural_network(self):
        network_object = self.network_class(**self.config)
        network_object.create_model()
        network_object.set_weights(self.weights)
        return network_object


class NeuralNetworkModel(ABC):
    def __init__(self, map_size, training_batch_size=100, training_epochs=10, inference_batch_size=400):
        self.map_size = map_size
        self.copy_setter_methods: List = []
        self.training_batch_size = training_batch_size
        self.inference_batch_size = inference_batch_size
        self.training_epochs = training_epochs
        self.model = None

    def load_latest_model(self):
        latest_model_file = get_latest_model()
        print("\n\nLoading model contained in file: %s\n\n" % (latest_model_file))
        self.model = tf.keras.models.load_model(latest_model_file)

    def load_model(self, model_name):
        return tf.keras.models.load_model(get_model_file_path(model_name))

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def get_skeleton(self):
        pass

    def get_copy(self):
        new_instance = self.get_skeleton()
        return new_instance.resurrect_neural_network()

    def add_synchronized_copy(self, copy):
        self.copy_setter_methods.append(copy)

    def save(self, model_name):
        self.model.save(get_model_file_path(model_name))

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

    def format_input(self, game_boards: List[np.ndarray], players=None):
        if players is not None:
            own_symbols = np.array(list(map(lambda player: player.get_symbol(), players)))
        else:
            own_symbols = np.array(1)
        own_symbols = own_symbols.reshape((-1, 1, 1))
        numeric_boards = np.array(game_boards)
        own_pieces = np.array(numeric_boards == own_symbols, dtype=np.float32)
        opponent_pieces = np.array(numeric_boards == -own_symbols, dtype=np.float32)
        numeric_representation = np.stack([own_pieces, opponent_pieces], axis=3)
        return numeric_representation

    def train(self, inputs, output_policies, output_values, validation_dataset):
        '''if validation_dataset is not None:
            validation_input = self.format_input(validation_dataset.board_states)
            validation_output_policies = np.array(validation_dataset.move_probabilities)
            validation_output_policies = validation_output_policies.reshape(validation_output_policies.shape[0], -1)
            validation_output_values = np.array(validation_dataset.rewards)
            validation_dataset = (validation_input, [validation_output_policies, validation_output_values])'''

        return self.model.fit(x=inputs, y=[output_policies, output_values], epochs=self.training_epochs, shuffle=True,
                              verbose=1, batch_size=self.training_batch_size)

    def predict(self, board_states, players):
        board_size = board_states[0].shape
        input = self.format_input(board_states, players)
        if len(input) == 0:
            print("biggus problemus")
        output_2d, value = self.model.predict(input, batch_size=self.inference_batch_size)
        output_2d = output_2d.reshape(-1, board_size[0], board_size[1])
        return output_2d, value


class PolicyValueNeuralNetwork(NeuralNetworkModel):
    def __init__(self, map_size, first_convolution_size=(9, 9), dropout=0.0, reg=1e-3,
                 training_epochs=10,
                 training_batch_size=16,
                 inference_batch_size=400):
        self.first_convolution_size = first_convolution_size
        self.dropout = dropout
        self.reg = reg
        super().__init__(map_size, training_batch_size, training_epochs, inference_batch_size)

    def get_skeleton(self):
        config = {"map_size": self.map_size,
                  "first_convolution_size": self.first_convolution_size,
                  "dropout": self.dropout,
                  "reg": self.reg,
                  "training_epochs": self.training_epochs,
                  "training_batch_size": self.training_batch_size}
        return NeuralNetworkSkeleton(self.__class__, config, self.get_weights())

    def create_model(self):
        input = Input(shape=self.map_size + (2,))
        conv_1 = Activation('relu')(
            BatchNormalization(axis=3)(Conv2D(32, kernel_size=self.first_convolution_size, padding='same',
                                              kernel_regularizer=l2(l2=self.reg))(input)))
        conv_2 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same',
                   kernel_regularizer=l2(l2=self.reg))(conv_1)))
        conv_3 = Activation('relu')(
            BatchNormalization(axis=3)(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                                              kernel_regularizer=l2(l2=self.reg))(conv_2)))
        flatten = Flatten()(conv_3)
        dense_1 = Dropout(self.dropout)(Activation('relu')(Dense(256, kernel_regularizer=l2(l2=self.reg))(flatten)))
        dense_2 = Dropout(self.dropout)(Activation('relu')(Dense(128, kernel_regularizer=l2(l2=self.reg))(dense_1)))
        policy = Dense(np.prod(self.map_size), activation='softmax', kernel_regularizer=l2(l2=self.reg))(dense_1)
        value = Dense(1, activation='tanh', kernel_regularizer=l2(l2=self.reg))(dense_2)

        model = Model(inputs=input, outputs=[policy, value])
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(learning_rate=0.001))
        self.model = model


class ResNetLike(NeuralNetworkModel):
    def __init__(self, map_size, network_depth=8, reg=0.000001, training_epochs=12, training_batch_size=16,
                 inference_batch_size=400):
        self.network_depth = network_depth
        self.reg = reg
        super().__init__(map_size, training_batch_size, training_epochs, inference_batch_size)

    def get_skeleton(self):
        config = {"map_size": self.map_size,
                  "network_depth": self.network_depth,
                  "reg": self.reg,
                  "training_epochs": self.training_epochs,
                  "training_batch_size": self.training_batch_size}
        return NeuralNetworkSkeleton(self.__class__, config, self.get_weights())

    def create_model(self):
        input = Input(shape=self.map_size + (2,))
        conv_1 = self.conv_layer(input, 64, (3, 3))
        current_network_end = conv_1
        for index in range(self.network_depth):
            current_network_end = self.identity_block(current_network_end, filters=[64, 64])

        policy = self.get_policy_head(current_network_end, self.map_size)
        value = self.get_value_head(current_network_end)

        model = Model(inputs=input, outputs=[policy, value])
        optimizer = Adam(learning_rate=0.003)
        # optimizer = SGD(learning_rate=0.01)
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=optimizer, loss_weights=[1, 1])
        self.model = model

    def get_value_head(self, feature_extractor):
        dim_reducer_conv = self.conv_layer(feature_extractor, 2, (1, 1))
        flatten = Flatten()(dim_reducer_conv)
        dense_1 = Dense(64, kernel_regularizer=l2(l2=self.reg))(flatten)
        value = Dense(1, activation='tanh', kernel_regularizer=l2(l2=self.reg),
                      name="value")(dense_1)
        return value

    def get_policy_head(self, feature_extractor, map_size):
        conv_1 = self.conv_layer(feature_extractor, 64, (3, 3))
        dimension_reducer_conv = self.conv_layer(conv_1, 2, (1, 1))
        flatten = Flatten()(dimension_reducer_conv)
        dense_1 = Dense(np.prod(map_size), activation='softmax', kernel_regularizer=l2(l2=self.reg),
                        name="policy")(flatten)
        return dense_1

    def identity_block(self, input, filters):
        conv_1 = self.conv_layer(input, filters[0], (3, 3))
        conv_2 = Conv2D(filters=filters[1], kernel_size=(3, 3), data_format="channels_first", padding='same'
                        , use_bias=False, activation='linear', kernel_regularizer=l2(l2=self.reg))(conv_1)
        batch_norm = BatchNormalization(axis=1)(conv_2)

        skip_connection = add([input, batch_norm])

        output = LeakyReLU()(skip_connection)
        return output

    def conv_layer(self, input, filters, kernel_size):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, data_format="channels_first", padding='same',
                      use_bias=False, activation='linear', kernel_regularizer=l2(self.reg))(input)

        batch_norm = BatchNormalization(axis=1)(conv)
        output = LeakyReLU()(batch_norm)

        return output
