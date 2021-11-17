import os
from abc import ABC, abstractmethod

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, BatchNormalization, Activation, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dropout, LeakyReLU
from tensorflow.python.keras.regularizers import l2


class NetworkModel(ABC):
    @abstractmethod
    def create_model(self, map_size):
        pass


class PolicyValueNetwork(NetworkModel):
    def __init__(self, first_convolution_size=(9, 9), dropout=0.0, reg=1e-3, training_epochs=10,
                 batch_size=16):
        self.first_convolution_size = first_convolution_size
        self.dropout = dropout
        self.training_epochs = training_epochs
        self.batch_size = batch_size
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
        value = Dense(1, activation='tanh', kernel_regularizer=l2(l2=self.reg))(dense_2)

        model = Model(inputs=input, outputs=[policy, value])
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(learning_rate=0.001))
        return model

    def train(self, model, train_inputs, train_outputs):
        return model.fit(x=train_inputs, y=train_outputs, epochs=self.training_epochs, shuffle=True,
                         verbose=1, batch_size=self.batch_size)


class ResNetLike(NetworkModel):
    def __init__(self, network_depth=8, reg=0.000001, training_epochs=12, batch_size=64):
        self.network_depth = network_depth
        self.reg = reg
        self.training_epochs = training_epochs
        self.batch_size = batch_size

    def create_model(self, map_size):
        input = Input(shape=map_size + (2,))
        conv_1 = self.conv_layer(input, 64, (3, 3))
        current_network_end = conv_1
        for index in range(self.network_depth):
            current_network_end = self.identity_block(current_network_end, filters=[64, 64])

        policy = self.get_policy_head(current_network_end, map_size)
        value = self.get_value_head(current_network_end)

        model = Model(inputs=input, outputs=[policy, value])
        optimizer = Adam(learning_rate=0.01)
        # optimizer = SGD(learning_rate=0.01)
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=optimizer, loss_weights=[1, 5])
        return model

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

    def train(self, model, train_inputs, train_outputs, **kwargs):
        if "epochs" not in kwargs:
            kwargs["epochs"] = self.training_epochs

        return model.fit(x=train_inputs, y=train_outputs, shuffle=True,
                         verbose=1, batch_size=self.batch_size, **kwargs)
