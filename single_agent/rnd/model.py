import tensorflow as tf
from tensorflow import layers
from tensorflow.distributions import Categorical


class ActorCriticModel(object):

    def __init__(self, obs, act_space):
        with tf.variable_scope('pi'):
            x = self._mlp(obs)
            logits = layers.dense(x, units=act_space.n)
            self.dist = Categorical(logits=logits)
        with tf.variable_scope('old_pi'):
            x = self._mlp(obs)
            logits = layers.dense(x, units=act_space.n)
            self.old_dist = Categorical(logits=logits)
        with tf.variable_scope('v'):
            x = self._mlp(obs)
            self.val = tf.squeeze(layers.dense(x, units=1))

    def _cnn(self, x):
        x = layers.conv2d(x, filters=32, kernel_size=8, strides=(4, 4), activation=tf.nn.tanh)
        x = layers.conv2d(x, filters=64, kernel_size=4, strides=(2, 2), activation=tf.nn.tanh)
        x = layers.conv2d(x, filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.tanh)
        x = layers.flatten(x)
        return layers.dense(x, units=512, activation=tf.nn.tanh)
    
    def _mlp(self, x):
        x = layers.dense(x, units=64, activation=tf.nn.tanh)
        x = layers.dense(x, units=64, activation=tf.nn.tanh)
        return x

    def output(self):
        return self.val, self.dist, self.old_dist


class RNDModel(object):

    def __init__(self, obs, rep_size):
        with tf.variable_scope('target'):
            self.target = self._mlp(obs, [128, 128, rep_size])
        with tf.variable_scope('predict'):
            self.predict = self._mlp(obs, [128, 128, 128, rep_size])

    def _mlp(self, x, hidden_sizes, activation=None, output_activation=None):
        for h in hidden_sizes[::-1]:
            x = layers.dense(x, units=h, activation=activation)
        return layers.dense(x, hidden_sizes[-1], activation=output_activation)

    def output(self):
        return self.target, self.predict
        
