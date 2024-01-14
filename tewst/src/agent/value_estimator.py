import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K



class ValueEstimator:

    def __init__(self, dim_state, leaning_rate=1e-3):
        self.dim_state = dim_state
        self.leaning_rate = leaning_rate
        self.height = dim_state[0]
        self.width = dim_state[1]
        self.build_network()
        self.compile()

    def build_network_old(self):
        nb_dense_1 = self.dim_state * 10
        nb_dense_3 = 5
        nb_dense_2 = int(np.sqrt(nb_dense_1 *
                                 nb_dense_3))

        l_input = Input(shape=(self.dim_state,),
                        name='input_state')
        l_dense_1 = tf.keras.layers.Dense(nb_dense_1,
                          activation='tanh',
                          name='hidden_1')(l_input)
        l_dense_2 = tf.keras.layers.Dense(nb_dense_2,
                          activation='tanh',
                          name='hidden_2')(l_dense_1)
        l_dense_3 = tf.keras.layers.Dense(nb_dense_3,
                          activation='tanh',
                          name='hidden_3')(l_dense_2)
        l_vs = tf.keras.layers.Dense(1, activation='linear',
                     name='Vs')(l_dense_3)

        self.model = Model(inputs=[l_input],
                           outputs=[l_vs])
        self.model.summary()

    def build_network(self):
        num_classes = 10

        input_layer = Input(shape = (self.height, self.width, 2))
        l_dense_1 = Conv2D(32, kernel_size = (3, 3), activation = 'relu')(input_layer)
        l_dense_2 = Conv2D(64, kernel_size = (3, 3), activation = 'relu')(input_layer)
        l_dense_3 = MaxPooling2D(pool_size=(2, 2))(l_dense_2)
        l_dense_4 = GlobalAveragePooling2D()(l_dense_3)  # Flatten の代わりに GlobalAveragePooling2D を使用
        l_mu = Dense(num_classes, activation = 'relu')(l_dense_4)
        l_log_var = Dense(num_classes, activation = 'softmax')(l_dense_4)
        self.model = Model(inputs = [input_layer], outputs = [l_mu, l_log_var])
        self.model.summary()

    def compile(self):
        self.state = tf.compat.v1.placeholder(
            tf.float32, shape=(None, self.height, self.width, 2))
        self.target = tf.compat.v1.placeholder(tf.float32,
                                     shape=(None, 1))

        self.state_value = self.model(self.state)

        self.loss = tf.math.squared_difference(
                self.state_value, self.target)
        self.loss = tf.reduce_mean(self.loss)

        optimizer = tf.compat.v1.train.AdamOptimizer(
            self.leaning_rate)
        self.minimize = optimizer.minimize(self.loss)

    def predict(self, sess, state):
        return sess.run(self.state_value,
                        {self.state: [state]})

    def update(self, sess, state, target):
        feed_dict = {
            self.state: state,
            self.target: target
        }
        _, loss = sess.run([self.minimize, self.loss],
                           feed_dict)
        return loss
