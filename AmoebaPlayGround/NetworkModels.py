import os
from abc import ABC, abstractmethod

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, BatchNormalization, Activation, Reshape, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.regularizers import l2


class NetworkModel(ABC):
    @abstractmethod
    def create_model(self, map_size):
        pass


class PolicyValueNetwork(NetworkModel):
    def __init__(self, first_convolution_size=(9, 9), dropout=0.0, reg=1e-3):
        self.first_convolution_size = first_convolution_size
        self.dropout = dropout
        self.reg = reg

    def create_model(self, map_size):
        input = Input(shape=map_size + (2,))
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
        policy = Dense(np.prod(map_size), activation='softmax', kernel_regularizer=l2(l2=self.reg))(dense_1)
        policy = Reshape(map_size)(policy)
        value = Dense(1, activation='tanh', kernel_regularizer=l2(l2=self.reg))(dense_2)

        model = Model(inputs=input, outputs=[policy, value])
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(lr=0.001))
        return model


class ResNetLike(NetworkModel):
    def __init__(self, network_depth=8, reg=1e-5):
        self.network_depth = network_depth
        self.reg = reg

    def create_model(self, map_size):
        input = Input(shape=map_size + (2,))
        conv_1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activity_regularizer=l2(self.reg))(
            input)
        batch_norm_1 = BatchNormalization()(conv_1)
        relu_1 = Activation('relu')(batch_norm_1)
        current_network_end = relu_1
        for index in range(self.network_depth):
            current_network_end = self.identity_block(current_network_end, filters=[64, 64])

        policy = self.get_policy_head(current_network_end, map_size)
        value = self.get_value_head(current_network_end)

        model = Model(inputs=input, outputs=[policy, value])
        optimizer = Adam(lr=0.001)
        # optimizer = SGD(lr=0.01)
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=optimizer)
        return model

    def get_value_head(self, feature_extractor):
        dimension_reducer_convolution = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                               activity_regularizer=l2(self.reg))(feature_extractor)
        value_norm = BatchNormalization()(dimension_reducer_convolution)
        dimension_reducer_activation = Activation('relu')(value_norm)
        flatten = Flatten()(dimension_reducer_activation)
        dense_1 = Dense(128, kernel_regularizer=l2(l2=self.reg))(flatten)
        dense_norm = BatchNormalization()(dense_1)
        dense_1_act = Activation('relu')(dense_norm)
        value = Dense(1, activation='tanh', kernel_regularizer=l2(l2=self.reg))(dense_1_act)
        return value

    def get_policy_head(self, feature_extractor, map_size):
        policy_conv_1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               activity_regularizer=l2(self.reg))(feature_extractor)
        policy_norm = BatchNormalization()(policy_conv_1)
        policy_activation = Activation('relu')(policy_norm)

        dimension_reducer_convolution = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                               activity_regularizer=l2(self.reg))(policy_activation)

        value_norm = BatchNormalization()(dimension_reducer_convolution)
        dimension_reducer_activation = Activation('relu')(value_norm)

        flatten = Flatten()(dimension_reducer_activation)
        dense_1 = Dense(np.prod(map_size), activation='softmax', kernel_regularizer=l2(l2=self.reg))(flatten)
        policy = Reshape(map_size)(dense_1)
        return policy

    def identity_block(self, input, filters):
        conv_1 = Conv2D(filters[0], kernel_size=(3, 3), strides=(1, 1), padding='same',
                        activity_regularizer=l2(self.reg))(input)
        batch_norm_1 = BatchNormalization()(conv_1)
        relu_1 = Activation('relu')(batch_norm_1)
        conv_2 = Conv2D(filters[1], kernel_size=(3, 3), strides=(1, 1), padding='same',
                        activity_regularizer=l2(self.reg))(relu_1)
        batch_norm_2 = BatchNormalization()(conv_2)
        skip_connection = Add()([batch_norm_2, input])
        relu_2 = Activation('relu')(skip_connection)
        return relu_2
