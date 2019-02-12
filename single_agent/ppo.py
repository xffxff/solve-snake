import time

import gym
import numpy as np
import tensorflow as tf
from tensorflow import layers
from tensorflow.distributions import Categorical

import snake_gym
from baselines.common.atari_wrappers import WarpFrame
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from utils.dqn_utils import PiecewiseSchedule
from utils.logx import EpochLogger
from gym.wrappers import TimeLimit

import os.path as osp 
from utils.checkpointer import get_latest_check_num


def create_env(n_env, seed, test=False):
    def make_env(rank):
        def _thunk():
            env = gym.make('Snake-rgb-v1')
            env.seed(seed + rank)
            env = LogWrapper(env, test, max_episode_steps=1000)
            env = WarpFrame(env)
            return env
        return _thunk
    env = SubprocVecEnv([make_env(i) for i in range(n_env)])
    return VecFrameStack(env, 2)


class LogWrapper(gym.Wrapper):

    def __init__(self, env, test=False, max_episode_steps=None):
        gym.Wrapper.__init__(self, env)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        self.test = test
        self.max_episode_steps = max_episode_steps

        self.ep_rew = 0.
        self.ep_len = 0
        self.t = 0

        self.foods = PiecewiseSchedule(
            [
                (0, 15),
                (2e4, 8),
                (8e4, 1)
            ],outside_value=1
        )

    def reset(self):
        if not self.test:
            self.env.set_foods(int(self.foods.value(self.t)))        
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.ep_rew += rew
        self.ep_len += 1
        self.t += 1
        if self.ep_len == self.max_episode_steps:
            done = True
        if done:
            info = {'ep_r': self.ep_rew, 'ep_len': self.ep_len, 'foods': int(self.foods.value(self.t))}
            self.ep_len, self.ep_rew = 0, 0.
        return obs, rew, done, info


class Buffer(object):

    def __init__(self, gamma, lam):
        self.gamma = gamma
        self.lam = lam
        self.obs_buf, self.act_buf, self.rew_buf, self.done_buf, self.val_buf = [], [], [], [], []

    def store(self, obs, act, rew, done, val):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.done_buf.append(done)
        self.val_buf.append(val)
    
    def get(self, last_val):
        self.val_buf.append(last_val)
        obs_buf = np.asarray(self.obs_buf, dtype=np.float32)
        act_buf = np.asarray(self.act_buf, dtype=np.int32)
        rew_buf = np.asarray(self.rew_buf, dtype=np.float32)
        done_buf = np.asarray(self.done_buf, dtype=np.bool)
        val_buf = np.asarray(self.val_buf, dtype=np.float32)
        adv_buf, ret_buf = np.zeros_like(rew_buf), np.zeros_like(rew_buf)
        last_gae_lam, last_ret = 0, last_val
        for i in reversed(range(len(self.rew_buf))):
            delta = rew_buf[i] + self.gamma * val_buf[i+1] * (1 - done_buf[i]) - val_buf[i]
            adv_buf[i] = last_gae_lam = delta + self.gamma * self.lam * (1 - done_buf[i]) * last_gae_lam
            ret_buf[i] = last_ret =  rew_buf[i] + self.gamma * last_ret * (1 - done_buf[i])
        obs_buf, act_buf, ret_buf, adv_buf = map(self.swap_and_flatten, (obs_buf, act_buf, ret_buf, adv_buf))
        self.obs_buf, self.act_buf, self.rew_buf, self.done_buf, self.val_buf = [], [], [], [], []
        adv_buf = (adv_buf - np.mean(adv_buf)) / np.std(adv_buf)
        return obs_buf, act_buf, ret_buf, adv_buf
    
    def swap_and_flatten(self, arr):
        shape = arr.shape
        return arr.swapaxes(0, 1).reshape(shape[0]*shape[1], *shape[2:])


class Net(object):

    def __init__(self, obs, act_space):
        with tf.variable_scope('pi'):
            x = self._cnn(obs)
            logits = layers.dense(x, units=act_space.n)
            self.dist = Categorical(logits=logits)
        with tf.variable_scope('old_pi'):
            x = self._cnn(obs)
            logits = layers.dense(x, units=act_space.n)
            self.old_dist = Categorical(logits=logits)
        with tf.variable_scope('v'):
            x = self._cnn(obs)
            self.val = tf.squeeze(layers.dense(x, units=1))

    def _cnn(self, x):
        x = layers.conv2d(x, filters=32, kernel_size=8, strides=(4, 4), activation=tf.nn.tanh)
        x = layers.conv2d(x, filters=64, kernel_size=4, strides=(2, 2), activation=tf.nn.tanh)
        x = layers.conv2d(x, filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.tanh)
        x = layers.flatten(x)
        return layers.dense(x, units=512, activation=tf.nn.tanh)
    
    def output(self):
        return self.val, self.dist, self.old_dist
    

class Agent(object):

    def __init__(self,
                 obs_space,
                 act_space,
                 clip_ratio=0.2,
                 ent_coef=0.01,
                 pi_lr=0.0001,
                 v_lr=0.0001):
        self.obs_space = obs_space
        self.act_space = act_space

        self._create_placeholders()
        self._create_network()

        self.act = self.dist.sample()

        self.pi = self.dist.prob(self.act_ph)
        self.old_pi = tf.stop_gradient(self.old_dist.prob(self.act_ph))

        self.kl = tf.reduce_mean(self.old_dist.kl_divergence(self.dist))
        self.entropy = tf.reduce_mean(self.dist.entropy())

        ratio = self.pi / self.old_pi
        min_adv = tf.where(self.adv_ph > 0, (1 + clip_ratio) * self.adv_ph, (1 - clip_ratio) * self.adv_ph)
        self.pi_loss = - tf.reduce_mean(tf.minimum(ratio * self.adv_ph, min_adv)) - ent_coef * self.entropy
        self.v_loss = tf.reduce_mean((self.ret_ph - self.val)**2)

        self.train_pi = tf.train.AdamOptimizer(pi_lr).minimize(self.pi_loss)
        self.train_v = tf.train.AdamOptimizer(v_lr).minimize(self.v_loss)

        self.pi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pi')
        self.old_pi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='old_pi')
        self.sync_old_pi_params_op = tf.group([tf.assign(old_params, params)\
                                                for old_params, params in zip(self.old_pi_params, self.pi_params)])


        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sync_old_pi_params()

        self.saver = tf.train.Saver(max_to_keep=3)
    
    def _create_placeholders(self):
        self.obs_ph = tf.placeholder(tf.float32, shape=[None] + list(self.obs_space.shape))
        self.act_ph = tf.placeholder(tf.int32, shape=[None, ])
        self.adv_ph = tf.placeholder(tf.float32, shape=[None, ])
        self.ret_ph = tf.placeholder(tf.float32, shape=[None, ])    

    def _create_network(self):
        obs_normal = tf.cast(self.obs_ph) / 255.0
        net = Net(obs_normal, self.act_space)
        self.val, self.dist, self.old_dist = net.output()
    
    def select_action(self, obs):
        act = self.sess.run(self.act, feed_dict={self.obs_ph: obs})
        return act

    def get_val(self, obs):
        val = self.sess.run(self.val, feed_dict={self.obs_ph: obs})
        return val

    def update_pi_params(self, feed_dict):
        _, pi_loss = self.sess.run([self.train_pi, self.pi_loss], feed_dict=feed_dict)
        return pi_loss

    def update_v_params(self, feed_dict):
        _, v_loss = self.sess.run([self.train_v, self.v_loss], feed_dict=feed_dict)
        return v_loss

    def sync_old_pi_params(self):
        self.sess.run(self.sync_old_pi_params_op)

    def get_kl(self, feed_dict):
        return self.sess.run([self.kl, self.entropy], feed_dict)

    def save_model(self, checkpoints_dir, epoch):
        self.saver.save(self.sess, osp.join(checkpoints_dir, 'tf_ckpt'), global_step=epoch)
    
    def load_model(self, checkpoints_dir):
        latest_model = get_latest_check_num(checkpoints_dir)
        self.saver.restore(self.sess, osp.join(checkpoints_dir, 'tf_ckpt-{}'.format(latest_model)))


class Runner(object):

    def __init__(self,
                 epochs,
                 n_env,
                 seed,
                 gamma=0.99,
                 lam=0.95,
                 train_epoch_len=500,
                 test_epoch_len=2000,
                 dtarg=0.01,
                 train_pi_iters=80,
                 train_v_iters=80,
                 logger_kwargs=dict()):

        self.epochs = epochs
        self.train_epoch_len = train_epoch_len
        self.test_epoch_len = test_epoch_len
        self.dtarg = dtarg
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.logger_kwargs = logger_kwargs

        self.checkpoints_dir = self.logger_kwargs['output_dir'] + '/checkpoints'
        
        tf.set_random_seed(seed)
        np.random.seed(seed)
        self.env = create_env(n_env, seed)

        self.obs = self.env.reset()

        obs_space = self.env.observation_space
        act_space = self.env.action_space

        self.agent = Agent(obs_space, act_space)
        self.buffer = Buffer(gamma, lam)
    
    def _collect_rollouts(self, logger):
        for step in range(self.train_epoch_len):
            acts = self.agent.select_action(self.obs)
            vals = self.agent.get_val(self.obs)
            logger.store(Val=vals)
            next_obs, rews, dones, infos = self.env.step(acts)
            self.buffer.store(self.obs, acts, rews, dones, vals)
            self.obs = next_obs
            for info in infos:
                if info.get('ep_r'):
                    logger.store(EpRet=info.get('ep_r'))
                    logger.store(EpLen=info.get('ep_len'))
                    logger.store(NFoods=info.get('foods'))
        last_val = self.agent.get_val(self.obs)
        return last_val

    def _run_train_phase(self, logger):
        start_time = time.time()
        last_val = self._collect_rollouts(logger)
        obs_buf, act_buf, ret_buf, adv_buf = self.buffer.get(last_val)
        feed_dict = {
            self.agent.obs_ph: obs_buf,
            self.agent.act_ph: act_buf,
            self.agent.ret_ph: ret_buf,
            self.agent.adv_ph: adv_buf,
        }

        for i in range(self.train_pi_iters):
            kl, entropy = self.agent.get_kl(feed_dict)
            logger.store(KL=kl, Entropy=entropy)
            if kl > 1.5 * self.dtarg:
                logger.log('Early stopping at step {} due to reaching max kl.'.format(i))
                break
            pi_loss = self.agent.update_pi_params(feed_dict)
            logger.store(PiLoss=pi_loss)
        for i in range(self.train_v_iters):
            v_loss = self.agent.update_v_params(feed_dict)
            logger.store(VLoss=v_loss)
        self.agent.sync_old_pi_params()
    
    def run_experiment(self):
        start_time = time.time()
        logger = EpochLogger(**self.logger_kwargs)
        for epoch in range(1, self.epochs + 1):
            self._run_train_phase(logger)
            self.agent.save_model(self.checkpoints_dir, epoch)
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', with_min_and_max=True)
            logger.log_tabular('NFoods', average_only=True)
            logger.log_tabular('Val', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('Entropy', average_only=True)
            logger.log_tabular('PiLoss', average_only=True)
            logger.log_tabular('VLoss', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

    def _run_test_phase(self, logger, render=True):
        env = create_env(1, 0, test=True)
        ep_r, ep_len = 0, 0
        obs = env.reset()
        for step in range(self.test_epoch_len):
            if render: env.render()
            act = self.agent.select_action(obs)
            next_obs, reward, done, info = env.step(act)
            time.sleep(0.1)
            ep_r += reward
            ep_len += 1
            obs = next_obs
            
            if done:
                logger.store(TestEpRet=ep_r, TestEpLen=ep_len)

                obs = env.reset()
                ep_r, ep_len = 0, 0

    def run_test_and_render(self):
        logger = EpochLogger()
        self.agent.load_model(self.checkpoints_dir)
        for epoch in range(1, self.epochs + 1):
            self._run_test_phase(logger)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--n_env', '-n', type=int, default=8)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    from utils.run_utils  import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, 'Snake-rgb-v1', args.seed)

    runner = Runner(args.epochs ,args.n_env, args.seed, logger_kwargs=logger_kwargs)
    if args.test:
        runner.run_test_and_render()
    else:
        runner.run_experiment()
