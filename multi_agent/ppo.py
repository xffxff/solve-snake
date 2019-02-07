
import tensorflow as tf
import numpy as np 
import gym
import snake_gym

from utils.wrappers import MultiWrapFrame, MultiFrameStack
from utils.subproc_vec_env import SubprocVecEnv

from tensorflow import layers
from tensorflow.distributions import Categorical 


def create_env(n_envs, seed):
    def make_env(rank):
        def _thunk():
            env = gym.make('MultiSnake-v0')
            env.seed(rank + seed)
            env = MultiWrapFrame(env)
            env = MultiFrameStack(env, 2)
            return env
        return _thunk
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    return env


class PPONet(object):

    def __init__(self,
                 obs,
                 act_n):
        with tf.variable_scope('v'):
            self.v = self._cnn(obs, 1)
        with tf.variable_scope('pi'):
            logits = self._cnn(obs, act_n)
            self.dist = Categorical(logits)
        with tf.variable_scope('old_pi'):
            logits = self._cnn(obs, act_n)
            self.old_dist = Categorical(logits)
    
    def _cnn(self, x, out_dim):
        x = layers.conv2d(x, filters=32, kernel_size=8, strides=(4, 4), activation=tf.nn.tanh)
        x = layers.conv2d(x, filters=64, kernel_size=4, strides=(3, 3), activation=tf.nn.tanh)
        x = layers.conv2d(x, filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.tanh)
        x = layers.flatten(x)
        x = layers.dense(x, units=512, activation=tf.nn.tanh)
        x = layers.dense(x, units=128, activation=tf.nn.tanh)
        return layers.dense(x, units=out_dim, activation=None)

    def output(self):
        return self.v, self.dist, self.old_dist


class PPOAgent(object):

    def __init__(self, 
                 act_n, 
                 obs_dim):
        
        self.act_n = act_n
        self.obs_dim = obs_dim

        self._create_placeholders()
        self.v, self.dist, self.old_dist = self._create_network()

        self.act = self.dist.sample()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def _create_placeholders(self):
        img_h, img_w, img_c = self.obs_dim
        self.obs_ph = tf.placeholder(tf.float32, shape=(None, img_h, img_w, img_c))
        self.act_ph = tf.placeholder(tf.int32, shape=(None, ))
    
    def _create_network(self):
        ppo_net = PPONet(self.obs_ph, self.act_n)
        v, dist, old_dist = ppo_net.output()
        return v, dist, old_dist

    def select_action(self, obs):
        act = self.sess.run(self.act, feed_dict={self.obs_ph: obs})
        return act


class Runner(object):

    def __init__(self, 
                 seed, 
                 n_env,
                 train_epoch_len=1000,
                 ):
        self.n_env = n_env
        self.train_epoch_len = train_epoch_len

        self.env = create_env(n_env, seed)
        tf.set_random_seed(seed)
        np.random.seed(seed)

        self.obs = self.env.reset()
        self.dones = [[False, False] for i in range(4)]

        obs_dim = [84, 84, 2]
        act_n = 4
        self.agent = PPOAgent(act_n, obs_dim)
    
    def _collect_rollouts(self, epoch_len):
        import pdb; pdb.set_trace()
        obs = self.obs.swapaxes(0, 1)
        act0 = self.agent.select_action(obs[0])
        act1 = self.agent.select_action(obs[1])
        act = list(zip(act0, act1))
        self.obs, rews, self.dones, info = self.env.step(act)
        print('hello')
        

    def _run_train_phase(self, epoch_len):
        self._collect_rollouts(epoch_len)

    def run_experiments(self):
        self._run_train_phase(self.train_epoch_len)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--n_env', '-n', type=int, default=4)
    parser.add_argument('--exp_name', '-e', type=str, default='multi-ppo')
    args = parser.parse_args()

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, '', args.seed)

    runner = Runner(args.seed, args.n_env)
    runner.run_experiments()