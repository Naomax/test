import numpy as np
import tensorflow as tf
import param
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, MaxPooling2D, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K

tf.compat.v1.disable_eager_execution()
class PolicyEstimator:

    def __init__(self,
                 dim_state,
                 dim_action,
                 leaning_rate=1e-3):
        self.dim_state = dim_state
        self.height = dim_state[0]
        self.width = dim_state[1]
        self.dim_action = dim_action
        self.leaning_rate = leaning_rate
        self.build_network()
        tf.compat.v1.disable_eager_execution()
        self.compile()

    def build_network_old(self):
        nb_dense_1 = self.dim_state * 10
        nb_dense_3 = self.dim_action * 10
        #nb_dense_2 = int(np.sqrt(nb_dense_1 *
        #                         nb_dense_3))
        

        print("dimstate ", self.dim_state)
        print("shape", (self.height, self.width, 2))
        l_input = Conv2D(64, (3, 3), activation = 'relu', input_shape = self.dim_state)
        #l_input = Input(shape=(self.dim_state,),
        #               name='input_state')
        l_dense_1 = Dense(nb_dense_1,
                          activation='tanh',
                          name='hidden_1')(l_input)
        l_dense_2 = Dense(nb_dense_2,
                          activation='tanh',
                          name='hidden_2')(l_dense_1)
        l_dense_3 = Dense(nb_dense_3,
                          activation='tanh',
                          name='hidden_3')(l_dense_2)
        l_mu = Dense(self.dim_action,
                     activation='tanh',
                     name='mu')(l_dense_3)
        l_log_var = Dense(self.dim_action,
                          activation='tanh',
                          name='log_var')(l_dense_3)

        self.model = Model(inputs=[l_input],
                           outputs=[l_mu, l_log_var])
        self.model.summary()

    def build_network(self):
        num_classes = 1

        input_layer = Input(shape = (self.height, self.width, 2))
        l_dense_1 = Conv2D(32, kernel_size = (3, 3), activation = 'relu')(input_layer)
        l_dense_2 = Conv2D(64, kernel_size = (3, 3), activation = 'relu')(input_layer)
        l_dense_3 = MaxPooling2D(pool_size=(2, 2))(l_dense_2)
        l_dense_4 = GlobalAveragePooling2D()(l_dense_3)  # Flatten の代わりに GlobalAveragePooling2D を使用
        l_mu = Dense(num_classes, activation = 'tanh')(l_dense_4)
        l_log_var = Dense(num_classes, activation = 'tanh')(l_dense_4)
        self.model = Model(inputs = [input_layer], outputs = [l_mu, l_log_var])
        self.model.summary()

    def compile(self):
        self.state = tf.compat.v1.placeholder(
            tf.float32, shape=(None, self.height, self.width, 2))
        self.action = tf.compat.v1.placeholder(
            tf.float32, shape=(None, 1))
        self.advantage = tf.compat.v1.placeholder(tf.float32,
                                        shape=(None, 1))

        self.mu, self.log_var = self.model(self.state)

        self.action_logprobs = self.logprob()
        self.loss = -self.action_logprobs * self.advantage
        self.loss = K.mean(self.loss)

        optimizer = tf.compat.v1.train.RMSPropOptimizer(
            self.leaning_rate)
        self.minimize = optimizer.minimize(self.loss)

    def logprob(self):
        action_logprobs = -0.5 * self.log_var
        action_logprobs += -0.5 \
            * K.square(self.action - self.mu) \
            / K.exp(self.log_var)
        return action_logprobs

    def predict(self, sess, state):
        mu, log_var = sess.run([self.mu, self.log_var],
                               {self.state: [state]})
        return np.random.normal(loc = mu[0][0], scale = np.sqrt(np.exp(log_var[0][0])))
        print("mu : ", mu)
        print("log_var : ", log_var)
        print("action : ", action)
        #mu, log_var = mu[0], log_var[0]
        #var = np.exp(log_var)
        #print("mu, log_var, var", mu, log_var, var)
        #action = np.random.normal(loc=mu,
        #                          scale=np.sqrt(var))
        return action

    def update(self, sess, state, action, advantage):
        feed_dict = {
            self.state: state,
            self.action: np.expand_dims(action, axis=1),
            #self.advantage: np.expand_dims(advantage, axis = 1),
            self.advantage: np.expand_dims(np.squeeze(advantage), axis=1),
        }
        _, loss = sess.run([self.minimize, self.loss],
                           feed_dict)
        return loss
