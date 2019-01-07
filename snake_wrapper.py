import gym
import numpy as np
from gym.spaces import Box


class ProcessFrame(gym.Wrapper):
    
    def __init__(self, env=None):
        super(ProcessFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(150, 150, 1))

    def _step(self, act):
        obs, rew, done, info = self.env.step(act)
        return self._process_frame(obs), rew, done, info

    def _reset(self):
        return self._process_frame(self.env.reset())
    
    def _process_frame(self, frame):
        img = (frame).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        return img.astype(np.uint8)

class FlattenFrame(gym.Wrapper):
    
    def __init__(self, env):
        super(FlattenFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(22500, ))
    
    def _step(self, act):
        obs, rew, done, info = self.env.step(act)
        return self._flatten_frame(obs), rew, done, info

    def _reset(self):
        return self._flatten_frame(self.env.reset())
    
    def _flatten_frame(self, frame):
        return np.reshape(frame, -1)


    


