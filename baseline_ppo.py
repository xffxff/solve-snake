import numpy as np
import gym
import gym_snake
from gym.spaces import Box, Discrete
from gym.utils import seeding
from gym.wrappers.time_limit import TimeLimit

from stable_baselines import bench, logger
from stable_baselines.common import set_global_seeds
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines.common.atari_wrappers import FrameStack


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
    

def train(env_id, num_timesteps, seed):
    """
    Train PPO2 model for Mujoco environment, for testing purposes

    :param env_id: (str) the environment id string
    :param num_timesteps: (int) the number of timesteps to run
    :param seed: (int) Used to seed the random generator.
    """
    def make_env(rank):
        def _thunk():
            env_out = gym.make(env_id)
            env_out.unit_size = 5
            env_out.snake_size = 5
            env_out.seed(seed + rank)
            env_out = TimeLimit(env_out, max_episode_steps=1000)
            env_out = EnvWrapper(env_out)
            # env_out = FrameStack(env_out, 2)
            # env_out = DummyVecEnv([lambda: env_out])
            env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
            return env_out
        return _thunk

    env = SubprocVecEnv([make_env(i) for i in range(8)])
    env = VecFrameStack(env, 3)

    print(env.reset().shape)

    set_global_seeds(seed)
    policy = CnnPolicy
    model = PPO2(policy=policy, env=env, n_steps=128, nminibatches=4, lam=0.95, gamma=0.99, noptepochs=4,
                 ent_coef=0.01, learning_rate=lambda f: f * 2.5e-4, cliprange=lambda f: f * 0.1, verbose=1)
    model.learn(total_timesteps=num_timesteps)

    return model, env

def main():
    logger.configure()
    model, env = train('snake-v0', num_timesteps=10000000, seed=0)
    model.save('ppo_snake')

main()