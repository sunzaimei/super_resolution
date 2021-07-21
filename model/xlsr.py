import tensorflow as tf
# from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, Activation, Concatenate
# from tensorflow.python.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Lambda, Activation, Concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from model.common import normalize, denormalize

layers = tf.keras.layers

class Xlsr:
    def __init__(self):
        self.initializer = tf.keras.initializers.HeNormal(seed=None)
        self.initializer.scale = 0.1

    def xlsr(self, num_gblocks=3, scale=3):
        x_in = Input(shape=(None, None, 3))
        # x = Lambda(normalize)(x_in)
        x = x_in
        conv_list = []
        side_block = Conv2D(16, (3, 3), strides=(1, 1), padding="same", kernel_initializer=self.initializer)(x)
        for _ in range(4):
            y = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation='relu',
                       kernel_initializer=self.initializer)(x)
            conv_list.append(y)
        main_block = Concatenate(axis=-1)(conv_list)
        main_block = Conv2D(32, (1, 1), padding='same', kernel_initializer=self.initializer)(main_block)
        for i in range(num_gblocks):
            main_block = self.gblock(main_block)
        block = Concatenate(axis=-1)([side_block, main_block])

        block = Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer=self.initializer)(block)
        block = Conv2D(3*scale**2, (3, 3), padding='same', kernel_initializer=self.initializer)(block)
        depth2spatial_layer = Lambda(lambda x: tf.nn.depth_to_space(x, scale, data_format='NHWC', name=None))
        block = depth2spatial_layer(block)

        def clip_relu(relu_input, max_value=255):
            return K.relu(relu_input, max_value=max_value)

        block = Lambda(function=clip_relu)(block)
        # block = Activation('relu')(block)
        # block = Lambda(denormalize)(block)
        return Model(x_in, block, name="xlsr")

    def gblock(self, x_in, filters=8):
        """
        :param x_in: shape should be 32 x 32 x 32. split into 4 groups with each (32, 32, 8)
        :param filters: conv2D no of filters
        :return:
        """
        x_list = tf.split(x_in, num_or_size_splits=4, axis=-1)
        conv_list = [layers.Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer=self.initializer)(x) for x in x_list]
        x = Concatenate(axis=-1)(conv_list)
        # x = layers.Conv2D(filters, (3, 3), padding='same', groups=4, activation='relu',
        #                   kernel_initializer=self.initializer)(x_in)
        x = Conv2D(32, (1, 1), padding='same')(x)
        return x
