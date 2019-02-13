import os.path as osp
import numpy as np
import tensorflow as tf

from model import ActorCriticModel, RNDModel
from utils.checkpointer import get_latest_check_num


class Agent(object):

    def __init__(self,
                 obs_space,
                 act_space,
                 clip_ratio=0.2,
                 ent_coef=0.01,
                 pi_lr=0.001,
                 v_lr=0.001,
                 rnd_lr=0.001):
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
        self.rnd_loss = tf.reduce_mean((tf.stop_gradient(self.target) - self.predict)**2)

        self.train_pi = tf.train.AdamOptimizer(pi_lr).minimize(self.pi_loss)
        self.train_v = tf.train.AdamOptimizer(v_lr).minimize(self.v_loss)
        self.train_rnd = tf.train.AdamOptimizer(rnd_lr).minimize(self.rnd_loss)

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
        actor_critic = ActorCriticModel(self.obs_ph, self.act_space)
        self.val, self.dist, self.old_dist = actor_critic.output()

        rnd = RNDModel(self.obs_ph, 64)
        self.target, self.predict = rnd.output()
    
    def select_action(self, obs):
        act = self.sess.run(self.act, feed_dict={self.obs_ph: obs})
        return act
    
    def get_intrinsic_reward(self, next_obs):
        target_rep, predict_rep = self.sess.run([self.target, self.predict], feed_dict={self.obs_ph: next_obs})
        intrinsic_reward = np.sum((target_rep - predict_rep)**2, axis=1)
        return intrinsic_reward

    def get_val(self, obs):
        val = self.sess.run(self.val, feed_dict={self.obs_ph: obs})
        return val

    def update_pi_params(self, feed_dict):
        _, pi_loss = self.sess.run([self.train_pi, self.pi_loss], feed_dict=feed_dict)
        return pi_loss

    def update_v_params(self, feed_dict):
        _, v_loss = self.sess.run([self.train_v, self.v_loss], feed_dict=feed_dict)
        return v_loss
    
    def update_rnd_params(self, feed_dict):
        _, rnd_loss = self.sess.run([self.train_rnd, self.rnd_loss], feed_dict=feed_dict)
        return rnd_loss

    def sync_old_pi_params(self):
        self.sess.run(self.sync_old_pi_params_op)

    def get_kl(self, feed_dict):
        return self.sess.run([self.kl, self.entropy], feed_dict)

    def save_model(self, checkpoints_dir, epoch):
        self.saver.save(self.sess, osp.join(checkpoints_dir, 'tf_ckpt'), global_step=epoch)
    
    def load_model(self, checkpoints_dir, model=None):
        if model is None:
            model = get_latest_check_num(checkpoints_dir)
        self.saver.restore(self.sess, osp.join(checkpoints_dir, 'tf_ckpt-{}'.format(model)))
