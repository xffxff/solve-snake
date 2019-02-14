import tensorflow as tf
from tensorflow import layers
from tensorflow.distributions import Categorical


class ActorCriticModel(object):

    def __init__(self, obs, act_space):
        with tf.variable_scope('pi'):
            x = self._cnn(obs)
            logits = layers.dense(x, units=act_space.n)
            self.dist = Categorical(logits=logits)
        with tf.variable_scope('old_pi'):
            x = self._cnn(obs)
            logits = layers.dense(x, units=act_space.n)
            self.old_dist = Categorical(logits=logits)

        share = self._cnn(obs)
        with tf.variable_scope('ext_v'):
            self.ext_val = tf.squeeze(layers.dense(share, units=1))
        with tf.variable_scope('int_v'):
            self.int_val = tf.squeeze(layers.dense(share, units=1))

    def _cnn(self, x):
        x = layers.conv2d(x, filters=32, kernel_size=8, strides=(4, 4), activation=tf.nn.tanh)
        x = layers.conv2d(x, filters=64, kernel_size=4, strides=(2, 2), activation=tf.nn.tanh)
        x = layers.conv2d(x, filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.tanh)
        x = layers.flatten(x)
        return layers.dense(x, units=512, activation=tf.nn.tanh)

    def output(self):
        return self.ext_val, self.int_val, self.dist, self.old_dist


class RNDModel(object):

    def __init__(self, obs):
        with tf.variable_scope('target'):
            self.target = self._target_net(obs)
        with tf.variable_scope('predict'):
            self.predict = self._predict_net(obs)

    def _target_net(self, x):
        x = layers.conv2d(x, filters=32, kernel_size=8, strides=(4, 4), activation=tf.nn.leaky_relu)
        x = layers.conv2d(x, filters=64, kernel_size=4, strides=(2, 2), activation=tf.nn.leaky_relu)
        x = layers.conv2d(x, filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.leaky_relu)
        x = layers.flatten(x)
        return layers.dense(x, units=512)        

    def _predict_net(self, x):
        x = layers.conv2d(x, filters=32, kernel_size=8, strides=(4, 4), activation=tf.nn.tanh)
        x = layers.conv2d(x, filters=64, kernel_size=4, strides=(2, 2), activation=tf.nn.tanh)
        x = layers.conv2d(x, filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.tanh)
        x = layers.flatten(x)
        x = layers.dense(x, units=512, activation=tf.nn.relu)
        x = layers.dense(x, units=512, activation=tf.nn.relu)
        return layers.dense(x, units=512)

    def output(self):
        return self.target, self.predict
        
