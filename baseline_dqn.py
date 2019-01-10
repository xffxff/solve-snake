
from stable_baselines.common.atari_wrappers import FrameStack
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN
from stable_baselines import bench, logger

import gym
import gym_snake
from gym.spaces import *
import numpy as np
from gym.wrappers.time_limit import TimeLimit


class EnvWrapper(gym.Wrapper):

    def __init__(self, env):
        super(EnvWrapper, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(75, 75, 1), dtype=np.uint8)
        self.action_space = Discrete(4)
    
    def _step(self, act):
        obs, rew, done, info = self.env.step(int(act))
        return self._process_frame(obs), rew, done, info
    
    def _reset(self):
        return self._process_frame(self.env.reset())

    def _process_frame(self, frame):
        img = (frame).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        img = np.reshape(img, [75, 75, 1])
        return img.astype(np.uint8)

logger.configure()

env = gym.make('snake-v0')
env.unit_size = 5
env.snake_size = 5
env = TimeLimit(env, max_episode_steps=1000)
env = EnvWrapper(env)
env = FrameStack(env, 3)
env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)

model = DQN(
    env=env,
    policy=CnnPolicy,
    learning_rate=1e-4,
    buffer_size=10000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    train_freq=4,
    learning_starts=10000,
    target_network_update_freq=1000,
    gamma=0.99,
    prioritized_replay=True,
    prioritized_replay_alpha=0.6,
    checkpoint_freq=10000,
    verbose=1,
)

model.learn(total_timesteps=10000000)
model.save("deepq_snake")

del model # remove to demonstrate saving and loading
