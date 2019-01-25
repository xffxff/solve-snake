
import time
import os.path as osp

import gym
import numpy as np
import tensorflow as tf
from gym.wrappers import TimeLimit
from utils.wrappers import WrapFrame, FrameStack, DistanceReward
from tensorflow import layers

import snake_gym
from utils.dqn_utils import *
from utils.logx import EpochLogger
from utils.checkpointer import get_latest_check_num


def create_atari_env(env_name):
    env = gym.make(env_name)
    env = DistanceReward(env)
    env = TimeLimit(env, max_episode_steps=1000)
    env = WrapFrame(env)
    return env


class DQNNet(object):

    def __init__(self, obs, act_n):
        out = layers.conv2d(obs, filters=32, kernel_size=8, strides=(4, 4), activation=tf.nn.relu)
        out = layers.conv2d(out, filters=64, kernel_size=4, strides=(2, 2), activation=tf.nn.relu)
        out = layers.conv2d(out, filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.relu)
        out = layers.flatten(out)
        out = layers.dense(out, units=512, activation=tf.nn.relu)
        self.out = layers.dense(out, units=act_n, activation=None)
    
    def network_output(self):
        return self.out


class DQNAgent(object):
    """An implementation of DQN agent."""

    def __init__(self,
                 obs_space,
                 act_space,
                 frame_stack,
                 gamma=0.99,
                 ):
        """Initialize the agent.

        Args:
            obs_space: gym.spaces, observation space.
            act_space: gym.spaces, action space.
            frame_stack: int, How many frames to stack as input to the net.
            gamma: float, Discount factor, (Always between 0 and 1.)
        """
        tf.logging.info('obs_space: {}'.format(obs_space))
        tf.logging.info('act_space: {}'.format(act_space))
        tf.logging.info('gamma: {}'.format(gamma))
        self.obs_space = obs_space
        self.act_space = act_space
        self.frame_stack = frame_stack

        self.act_n = self.act_space.n

        self._create_placeholders()
        self._create_network()

        q_act = tf.reduce_sum(tf.one_hot(self.act_ph, depth=self.act_n) * self.q_acts, axis=1)
        y = self.rew_ph + (1 - self.done_ph) * gamma * tf.reduce_max(self.q_targ_acts, axis=1)
        y = tf.stop_gradient(y)
        self.loss = tf.reduce_mean(huber_loss(y - q_act))

        main_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main')
        target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')

        self.train_op = tf.train.AdamOptimizer(self.lr_ph).minimize(self.loss)

        self.update_target_op = tf.group([tf.assign(target_var, main_var) \
                                for target_var, main_var in zip(target_vars, main_vars)])

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.update_target_op)

        self.saver = tf.train.Saver(max_to_keep=3)

    def _create_placeholders(self):
        img_h, img_w, img_c = self.obs_space.shape
        input_shape = (img_h, img_w, img_c * self.frame_stack)
        self.obs_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
        self.act_ph = tf.placeholder(tf.int32, [None])
        self.rew_ph = tf.placeholder(tf.float32, [None])
        self.next_obs_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
        self.done_ph = tf.placeholder(tf.float32, [None])
        self.lr_ph = tf.placeholder(tf.float32, None)

    def _create_network(self):
        with tf.variable_scope('main'):
            obs_float = tf.cast(self.obs_ph, tf.float32) / 255.0
            net = DQNNet(obs_float, self.act_n)
            self.q_acts = net.network_output()
        with tf.variable_scope('target'):
            next_obs_float = tf.cast(self.next_obs_ph, tf.float32) / 255.0
            net = DQNNet(next_obs_float, self.act_n)
            self.q_targ_acts = net.network_output()

    def select_action(self, obs):
        q_acts = self.sess.run(self.q_acts, feed_dict={self.obs_ph: obs})
        return np.argmax(q_acts)

    def train_q(self, feed_dict):
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss
    
    def update_target(self, feed_dict):
        self.sess.run(self.update_target_op, feed_dict=feed_dict)

    def save_model(self, checkpoints_dir, epoch):
        self.saver.save(self.sess, osp.join(checkpoints_dir, 'tf_ckpt'), global_step=epoch)

    def load_model(self, checkpoints_dir):
        latest_model = get_latest_check_num(checkpoints_dir)
        self.saver.restore(self.sess, osp.join(checkpoints_dir, 'tf_ckpt-{}'.format(latest_model)))


class DQNRunner(object):

    def __init__(self,
                 env_name, 
                 seed,
                 epochs=500,
                 train_epoch_len=10000,
                 test_epoch_len=2000,
                 start_learn=50000,
                 learning_freq=4,
                 target_update_freq=10000,
                 buffer_size=int(1e6),
                 batch_size=32,
                 frame_stack=2,
                 logger_kwargs=dict(),
                 ):
        """Initialize the Runner object.

        Args: 
            env_name: str, Name of the environment.
            seed: int, Seed of random number generators.
            epochs: int, Number of epochs to run and train agent.
            train_epoch_len: int, Number of steps of interactions (state-action pairs)
                for the agent and the environment in each training epoch.
            start_learn: int, After how many environment steps to start replaying experiences.
            learning_freq: int, How many steps of environment to take between every experience replay.
            target_update_freq: int, How many experience replay rounds (not steps!) to perform between
                each update to the target Q network.
            buffer_size: int, How many memories to store in the replay buffer.
            batch_size: int, How many transitions to sample each time experience is replayed.
            frame_stack: int, How many frames to stack as input to the net.
        """
        tf.logging.info('env_name: {}'.format(env_name))
        tf.logging.info('seed: {}'.format(seed))
        tf.logging.info('epochs: {}'.format(epochs))
        tf.logging.info('train_epoch_len: {}'.format(train_epoch_len))
        tf.logging.info('start_learn: {}'.format(start_learn))
        tf.logging.info('learning_freq: {}'.format(learning_freq))
        tf.logging.info('target_update_freq: {}'.format(target_update_freq))
        tf.logging.info('buffer_size: {}'.format(buffer_size))
        tf.logging.info('batch_size: {}'.format(batch_size))
        tf.logging.info('frame_stack: {}'.format(frame_stack))
        self.env = create_atari_env(env_name)
        self.epochs = epochs
        self.train_epoch_len = train_epoch_len
        self.test_epoch_len = test_epoch_len
        self.start_learn = start_learn
        self.learning_freq = learning_freq
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.logger_kwargs = logger_kwargs

        self.checkpoints_dir = self.logger_kwargs['output_dir'] + '/checkpoints'

        tf.set_random_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        obs_space = self.env.observation_space
        act_space = self.env.action_space

        self.max_ep_len = self.env.spec.timestep_limit

        self.obs = self.env.reset()
        self.ep_len, self.ep_r = 0, 0
        self.t = 0
        self.learning_step = 0

        self.exploration = PiecewiseSchedule(
            [
                (0, 1.0),
                (1e6, 0.1),
                (2e6, 0.01)
            ], outside_value=0.01,
        )

        self.lr_schedule = PiecewiseSchedule(
            [
            (0, 1e-4),
            (2e6, 5e-5)
            ], outside_value=5e-5,
        )

        self.replay_buffer = ReplayBuffer(buffer_size, frame_stack, lander=False)
        self.agent = DQNAgent(obs_space, act_space, frame_stack)

    def _run_one_step(self, logger):
        idx = self.replay_buffer.store_frame(self.obs)
        epsilon = self.exploration.value(self.t)
        if np.random.random() < epsilon:
            act = self.env.action_space.sample()
        else:
            act = self.agent.select_action(self.replay_buffer.encode_recent_observation()[None, :])
        next_obs, rew, done, info = self.env.step(act)
        self.ep_len += 1
        self.ep_r += rew
        self.t += 1
        done = False if self.ep_len == self.max_ep_len else done
        self.replay_buffer.store_effect(idx, act, rew, done)
        self.obs = next_obs
        if done or self.ep_len == self.max_ep_len:
            logger.store(EpRet=self.ep_r, EpLen=self.ep_len)
            self.obs = self.env.reset()
            self.ep_len, self.ep_r = 0, 0

    def _train_one_step(self, logger):
        if (self.t > self.start_learn and \
            self.t % self.learning_freq == 0 and \
            self.replay_buffer.can_sample(self.batch_size)):
            obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = self.replay_buffer.sample(self.batch_size)
            lr = self.lr_schedule.value(self.t)
            feed_dict = {
                self.agent.obs_ph: obs_batch,
                self.agent.act_ph: act_batch,
                self.agent.rew_ph: rew_batch,
                self.agent.next_obs_ph: next_obs_batch,
                self.agent.done_ph: done_batch,
                self.agent.lr_ph: lr,
            }
            loss = self.agent.train_q(feed_dict)
            logger.store(Loss=loss)
            if self.learning_step % self.target_update_freq == 0:
                self.agent.update_target(feed_dict)
            self.learning_step += 1
            
    def _run_train_phase(self, logger):
        for step in range(self.train_epoch_len):
            self._run_one_step(logger)
            self._train_one_step(logger)
    
    def run_test_phase(self, epoch_len, logger, render=False):
        """Run test phase.

        Args:
            epoch_len: int, Number of steps of interaction (state-action pairs)
                for the agent and the environment in each training epoch.
            logger: object, Object to store the information.
        """

        ep_r, ep_len = 0, 0
        obs = self.env.reset()
        for step in range(epoch_len):
            if render: self.env.render()
            self.replay_buffer.store_frame(obs)
            act = self.agent.select_action(self.replay_buffer.encode_recent_observation()[None, :])
            next_obs, reward, done, info = self.env.step(act)
            ep_r += reward
            ep_len += 1
            obs = next_obs
            
            if done or ep_len == self.max_ep_len:
                logger.store(TestEpRet=ep_r, TestEpLen=ep_len)

                obs = self.env.reset()
                ep_r, ep_len = 0, 0

    def run_experiment(self):
        logger = EpochLogger(**self.logger_kwargs)
        start_time = time.time()
        for epoch in range(1, self.epochs + 1):
            self._run_train_phase(logger)
            self.agent.save_model(self.checkpoints_dir, epoch)
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            try:
                logger.log_tabular('Loss', average_only=True)
            except:
                logger.log_tabular('Loss', 0)
            logger.log_tabular('LearningRate', self.lr_schedule.value(self.t))
            logger.log_tabular('Exploration', self.exploration.value(self.t))
            logger.log_tabular('TotalEnvInteracts', epoch * self.train_epoch_len)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

    def run_test_and_render(self):
        """Load the saved model and test it."""
        logger = EpochLogger()
        self.agent.load_model(self.checkpoints_dir)
        for epoch in range(self.epochs):
            self.run_test_phase(self.test_epoch_len, logger, render=True)
            logger.log_tabular('Epoch', epoch+1)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Snake-v0')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(exp_name='dqn', env_name=args.env_name, seed=args.seed)

    tf.logging.set_verbosity(tf.logging.INFO)
    
    runner = DQNRunner(args.env_name, args.seed, logger_kwargs=logger_kwargs)

    if args.test:
        runner.run_test_and_render()
    else:
        runner.run_experiment()

