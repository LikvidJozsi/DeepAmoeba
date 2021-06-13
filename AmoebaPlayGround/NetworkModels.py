from abc import ABC, abstractmethod
import numpy as np
from tensorflow.python.keras import regularizers

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, BatchNormalization, Activation, Reshape, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dropout

class NetworkModel(ABC):
    @abstractmethod
    def create_model(self, map_size):
        pass

class PolicyValueNetwork(NetworkModel):
    def __init__(self, first_convolution_size=(9, 9), dropout=0.4):
        self.first_convolution_size = first_convolution_size
        self.dropout = dropout

    def create_model(self, map_size):
        input = Input(shape=map_size + (2,))
        conv_1 = Activation('relu')(
            BatchNormalization(axis=3)(Conv2D(32, kernel_size=self.first_convolution_size, padding='same')(input)))
        conv_2 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(conv_1)))
        conv_3 = Activation('relu')(
            BatchNormalization(axis=3)(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(conv_2)))
        flatten = Flatten()(conv_3)
        dense_1 = Dropout(self.dropout)(Activation('relu')(Dense(256, activation='relu')(flatten)))
        dense_2 = Dropout(self.dropout)(Activation('relu')(Dense(128, activation='relu')(dense_1)))
        policy = Dense(np.prod(map_size), activation='softmax')(dense_1)
        policy = Reshape(map_size)(policy)
        value = Dense(1, activation='tanh')(dense_2)

        model = Model(inputs=input, outputs=[policy, value])
        model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(lr=0.001))
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