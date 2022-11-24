import copy
import glob
import io
import os
from abc import ABC
from typing import List

import h5py
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, BatchNormalization, Activation, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dropout, LeakyReLU
from tensorflow.python.keras.regularizers import l2
from AmoebaPlayGround.Training.TrainingSampleGenerator import TrainingSampleCollection, TrainingDatasetGenerator

models_folder = '../Models/'

tf.config.optimizer.set_jit(True)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_latest_model():
    list_of_files = glob.glob(os.path.join(models_folder, '*.h5'))
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def get_model_file_path(model_name):
    return os.path.join(models_folder, model_name + '.h5')


# State machine diagram of a NeuralNetworkModel:
#      created
#         |
#         | create_model or load_model or load_latest_model
#         |  ⌜--------------⌝
#         v  |              | load_weights
#  unpacked/compiled <------⌟
#        ^  | get_packed_copy
# unpack |  v
#       packed
#
#  created state  is the initial state with no initialized model (model is None and serialized model is None)
#  unpacked/compiled state has its own compiled tensorflow model (model exists serialized_model is None)
#  packed state has all the information but is not compiled, so it can be transported between processes (model is None and serialized_model exists)
class NeuralNetworkModel(ABC):
    def __init__(self, training_dataset_max_size, training_batch_size=100, training_epochs=10,
                 inference_batch_size=400):
        self.training_batch_size = training_batch_size
        self.inference_batch_size = inference_batch_size
        self.training_epochs = training_epochs
        self.training_dataset_max_size = training_dataset_max_size
        self.model = None
        self.serialized_model = None

    # this is used to get a copy of a model that can be transported between processes,
    # and can be recompiled using unpack
    def get_packed_copy(self):
        if self.model is None:
            raise Exception("the model needs to be unpacked/compiled to be packed")
        copied_model = copy.copy(self)
        copied_model.model = None
        h5_object = io.BytesIO()
        with h5py.File(h5_object, "w") as f:
            self.model.save(f)
        copied_model.serialized_model = h5_object
        return copied_model

    # use to recompile a packed model, after which it can be used
    def unpack(self):
        if self.serialized_model is None:
            raise Exception("you can only unpack a packed model")
        with h5py.File(self.serialized_model) as f:
            self.model = tf.keras.models.load_model(f)
        self.serialized_model = None

    # use if keeping the hyperparameters of the current compiled model object while getting the
    # weights of another(compatible) model is desired
    def load_weights(self, model_name):
        if self.model is None:
            raise Exception("the model needs to be unpacked/compiled to load weights")
        loaded_model = tf.keras.models.load_model(get_model_file_path(model_name), compile=False)
        self.model.set_weights(loaded_model.get_weights())

    # use if the most recent saved model is needed (along with its hyperparameters, it will compile the loaded model
    def load_latest_model(self):
        latest_model_file = get_latest_model()
        print("\n\nLoading model contained in file: %s\n\n" % (latest_model_file))
        self.model = tf.keras.models.load_model(latest_model_file)

    # load and compile any saved model
    def load_model(self, model_name):
        self.model = tf.keras.models.load_model(get_model_file_path(model_name))

    def save_model(self, model_name):
        self.model.save(get_model_file_path(model_name))

    def get_copy(self):
        new_instance = self.get_packed_copy()
        new_instance.unpack()
        return new_instance

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

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

    def train(self, dataset_generator: TrainingDatasetGenerator,
              validation_dataset: TrainingSampleCollection = None):
        print('number of training samples: ' + str(dataset_generator.get_sample_count()))
        inputs, output_policies, output_values = dataset_generator.get_dataset(self.training_dataset_max_size)
        output_policies = output_policies.reshape(output_policies.shape[0], -1)
        return self.model.fit(x=inputs, y=[output_policies, output_values], epochs=self.training_epochs, shuffle=True,
                              verbose=1, batch_size=self.training_batch_size)

    # TODO this is amoeba specific, refactor
    def predict(self, board_states, players):
        board_size = board_states[0].shape
        input = self.format_input(board_states, players)
        if len(input) == 0:
            print("biggus problemus")
        output_2d, value = self.model.predict(input, batch_size=self.inference_batch_size)
        output_2d = output_2d.reshape(-1, board_size[0], board_size[1])
        return output_2d, value


class PolicyValueNeuralNetwork(NeuralNetworkModel):
    def __init__(self, training_epochs=10,
                 training_batch_size=16,
                 inference_batch_size=400,
                 training_dataset_max_size=600000):
        super().__init__(training_dataset_max_size, training_batch_size, training_epochs,
                         inference_batch_size)

    def create_model(self, map_size, first_convolution_size=(9, 9), dropout=0.0, reg=1e-3, ):
        input = Input(shape=map_size + (2,))
        conv_1 = Activation('relu')(
            BatchNormalization(axis=3)(Conv2D(32, kernel_size=first_convolution_size, padding='same',
                                              kernel_regularizer=l2(l2=reg))(input)))
        conv_2 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same',
                   kernel_regularizer=l2(l2=reg))(conv_1)))
        conv_3 = Activation('relu')(
            BatchNormalization(axis=3)(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                                              kernel_regularizer=l2(l2=reg))(conv_2)))
        flatten = Flatten()(conv_3)
        dense_1 = Dropout(dropout)(Activation('relu')(Dense(256, kernel_regularizer=l2(l2=reg))(flatten)))
        dense_2 = Dropout(dropout)(Activation('relu')(Dense(128, kernel_regularizer=l2(l2=reg))(dense_1)))
        policy = Dense(np.prod(map_size), activation='softmax', kernel_regularizer=l2(l2=reg))(dense_1)
        value = Dense(1, activation='tanh', kernel_regularizer=l2(l2=reg))(dense_2)

        model = Model(inputs=input, outputs=[policy, value])
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(learning_rate=0.001))
        self.model = model


class ResNetLike(NeuralNetworkModel):
    def __init__(self, training_epochs=12, training_batch_size=16,
                 inference_batch_size=400, training_dataset_max_size=600000):
        super().__init__(training_dataset_max_size, training_batch_size, training_epochs,
                         inference_batch_size)

    def create_model(self, map_size, network_depth=8, reg=0.000001, learning_rate=0.003):
        input = Input(shape=map_size + (2,))
        conv_1 = self.conv_layer(input, 64, (3, 3), reg)
        current_network_end = conv_1
        for index in range(network_depth):
            current_network_end = self.identity_block(current_network_end, filters=[64, 64], reg=reg)

        policy = self.get_policy_head(current_network_end, map_size, reg)
        value = self.get_value_head(current_network_end, reg)

        model = Model(inputs=input, outputs=[policy, value])
        optimizer = Adam(learning_rate=learning_rate)
        # optimizer = SGD(learning_rate=0.01)
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=optimizer, loss_weights=[1, 1])
        self.model = model

    def get_value_head(self, feature_extractor, reg):
        dim_reducer_conv = self.conv_layer(feature_extractor, 2, (1, 1), reg)
        flatten = Flatten()(dim_reducer_conv)
        dense_1 = Dense(64, kernel_regularizer=l2(l2=reg))(flatten)
        value = Dense(1, activation='tanh', kernel_regularizer=l2(l2=reg),
                      name="value")(dense_1)
        return value

    def get_policy_head(self, feature_extractor, map_size, reg):
        conv_1 = self.conv_layer(feature_extractor, 64, (3, 3), reg)
        dimension_reducer_conv = self.conv_layer(conv_1, 2, (1, 1), reg)
        flatten = Flatten()(dimension_reducer_conv)
        dense_1 = Dense(np.prod(map_size), activation='softmax', kernel_regularizer=l2(l2=reg),
                        name="policy")(flatten)
        return dense_1

    def identity_block(self, input, filters, reg):
        conv_1 = self.conv_layer(input, filters[0], (3, 3), reg)
        conv_2 = Conv2D(filters=filters[1], kernel_size=(3, 3), data_format="channels_last", padding='same'
                        , use_bias=False, activation='linear', kernel_regularizer=l2(l2=reg))(conv_1)
        batch_norm = BatchNormalization(axis=1)(conv_2)

        skip_connection = add([input, batch_norm])

        output = LeakyReLU()(skip_connection)
        return output

    def conv_layer(self, input, filters, kernel_size, reg):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, data_format="channels_last", padding='same',
                      use_bias=False, activation='linear', kernel_regularizer=l2(reg))(input)

        batch_norm = BatchNormalization(axis=1)(conv)
        output = LeakyReLU()(batch_norm)

        return output


class ConstantModel:

    def __init__(self, batch_size):
        self.inference_batch_size = batch_size

    def predict(self, board_states, players):
        board_shape = board_states[0].shape
        uniform_policy = np.ones((len(board_states),) + board_shape) / np.prod(board_states[0].shape)
        for i in range(uniform_policy.shape[0]):
            uniform_policy[i] = uniform_policy[i] * 0.9 + 0.1 * np.random.dirichlet(
                [0.1] * np.prod(board_shape)).reshape(board_shape)
        return uniform_policy, np.zeros(len(board_states))

    def get_packed_copy(self):
        return self

    def unpack(self):
        pass

    def get_weights(self):
        return None

    def set_weights(self, weights):
        pass
