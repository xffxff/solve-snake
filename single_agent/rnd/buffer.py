import numpy as np
from utils.statistics_scalar import RewardForwardFilter, RunningMeanStd


class Buffer(object):

    def __init__(self, gamma, lam):
        self.gamma = gamma
        self.lam = lam
        self.obs_buf, self.act_buf, self.done_buf = [], [], []
        self.ext_rew_buf, self.int_rew_buf, self.ext_val_buf, self.int_val_buf = [], [], [], []
        self.rew_rms = RunningMeanStd()
        self.rew_discount = RewardForwardFilter(0.99)

    def store(self, obs, act, ext_rew, done, ext_val, int_val, int_rew):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.ext_rew_buf.append(ext_rew)
        self.int_rew_buf.append(int_rew)
        self.done_buf.append(done)
        self.ext_val_buf.append(ext_val)
        self.int_val_buf.append(int_val)
    
    def get(self, last_ext_val, last_int_val):
        self.ext_val_buf.append(last_ext_val)
        self.int_val_buf.append(last_int_val)
        obs_buf = np.asarray(self.obs_buf, dtype=np.float32)
        act_buf = np.asarray(self.act_buf, dtype=np.int32)
        ext_rew_buf = np.asarray(self.ext_rew_buf, dtype=np.float32)
        int_rew_buf = np.asarray(self.int_rew_buf, dtype=np.float32)
        done_buf = np.asarray(self.done_buf, dtype=np.bool)
        ext_val_buf = np.asarray(self.ext_val_buf, dtype=np.float32)
        int_val_buf = np.asarray(self.int_val_buf, dtype=np.float32)

        total_rew_per_env = np.array([self.rew_discount.update(rew_per_step) for rew_per_step in int_rew_buf])
        mean, std, count = np.mean(total_rew_per_env), np.std(total_rew_per_env), len(total_rew_per_env)
        self.rew_rms.update_from_moments(mean, std**2, count)
        int_rew_buf /= np.sqrt(self.rew_rms.var)
        # print(int_rew_buf)
        # self.rew_rms.update(int_rew_buf) 
        # rew_buf = ext_rew_buf + int_rew_buf

        ext_adv_buf, ext_ret_buf = np.zeros_like(ext_rew_buf), np.zeros_like(ext_rew_buf)
        int_adv_buf, int_ret_buf = np.zeros_like(int_rew_buf), np.zeros_like(int_rew_buf)
        ext_last_gae_lam, ext_last_ret = 0, last_ext_val
        int_last_gae_lam, int_last_ret = 0, last_int_val
        for i in reversed(range(len(self.ext_rew_buf))):
            # ext_delta = ext_rew_buf[i] + self.gamma * ext_val_buf[i+1] * (1 - done_buf[i]) - ext_val_buf[i]
            # ext_adv_buf[i] = ext_last_gae_lam = ext_delta + self.gamma * self.lam * (1 - done_buf[i]) * ext_last_gae_lam
            # ext_ret_buf[i] = ext_last_ret =  ext_rew_buf[i] + self.gamma * ext_last_ret * (1 - done_buf[i])

            # int_delta = int_rew_buf[i] + self.gamma * int_val_buf[i+1] * (1 - done_buf[i]) - int_val_buf[i]
            # int_adv_buf[i] = int_last_gae_lam = int_delta + self.gamma * self.lam * (1 - done_buf[i]) * int_last_gae_lam
            # int_ret_buf[i] = int_last_ret =  int_rew_buf[i] + self.gamma * int_last_ret * (1 - done_buf[i])

            ext_delta = ext_rew_buf[i] + self.gamma * ext_val_buf[i+1] - ext_val_buf[i]
            ext_adv_buf[i] = ext_last_gae_lam = ext_delta + self.gamma * self.lam * ext_last_gae_lam
            ext_ret_buf[i] = ext_last_ret =  ext_rew_buf[i] + self.gamma * ext_last_ret

            int_delta = int_rew_buf[i] + self.gamma * int_val_buf[i+1] - int_val_buf[i]
            int_adv_buf[i] = int_last_gae_lam = int_delta + self.gamma * self.lam * int_last_gae_lam
            int_ret_buf[i] = int_last_ret =  int_rew_buf[i] + self.gamma * int_last_ret

        obs_buf, act_buf, ext_ret_buf, ext_adv_buf, int_ret_buf, int_adv_buf \
                = map(self.swap_and_flatten, (obs_buf, act_buf, ext_ret_buf, ext_adv_buf, int_ret_buf, int_adv_buf))
        self.obs_buf, self.act_buf, self.done_buf = [], [], []
        self.ext_rew_buf, self.int_rew_buf, self.ext_val_buf, self.int_val_buf = [], [], [], []
        # print(int_ret_buf)

        adv_buf = ext_adv_buf + int_adv_buf 
        adv_buf = (adv_buf - np.mean(adv_buf)) / np.std(adv_buf)

        state_buf = obs_buf[:, :, :, -1][:, :, :, None]
        return obs_buf, act_buf, ext_ret_buf, int_ret_buf, adv_buf, state_buf
    
    def swap_and_flatten(self, arr):
        shape = arr.shape
        return arr.swapaxes(0, 1).reshape(shape[0]*shape[1], *shape[2:])
