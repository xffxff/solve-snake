
from collections import deque

import cv2
import gym
import numpy as np
from gym import spaces
from gym.wrappers import TimeLimit


class WrapFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.

        :param env: (Gym Environment) the environment
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1),
                                            dtype=env.observation_space.dtype)

    def observation(self, frame):
        """
        returns the current observation from a frame

        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]
    
    def set_foods(self, n):
        self.env.set_foods(n)


class MultiWrapFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.

        :param env: (Gym Environment) the environment
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.n_snakes = self.env.n_snakes
        self.observation_space = spaces.Tuple([spaces.Box(low=0, high=255, shape=(self.height, self.width, 1),
                                            dtype=env.observation_space.spaces[0].dtype) for i in range(self.env.n_snakes)])

    def observation(self, frames):
        """
        returns the current observation from a frame

        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        observation = []
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            observation.append(frame[:, :, None])
        return observation
    
    def set_foods(self, n):
        self.env.set_foods(n)

    def set_snakes(self, n):
        self.env.set_snakes(n)


class FrameStack(gym.Wrapper):
    def __init__(self, env, n_frames):
        """Stack n_frames last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        stable_baselines.common.atari_wrappers.LazyFrames

        :param env: (Gym Environment) the environment
        :param n_frames: (int) the number of frames to stack
        """
        gym.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * n_frames),
                                            dtype=env.observation_space.dtype)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.n_frames
        return np.concatenate(self.frames, axis=2)

    def set_foods(self, n):
        self.env.set_foods(n)
    


class MultiFrameStack(gym.Wrapper):
    def __init__(self, env, n_frames):
        """Stack n_frames last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        stable_baselines.common.atari_wrappers.LazyFrames

        :param env: (Gym Environment) the environment
        :param n_frames: (int) the number of frames to stack
        """
        gym.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.n_snakes = self.env.n_snakes
        self.frames = [deque([], maxlen=n_frames) for i in range(self.env.n_snakes)]
        shp = env.observation_space.spaces[0].shape
        self.observation_space = spaces.Tuple([spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * n_frames),
                                            dtype=env.observation_space.dtype) for i in range(self.env.n_snakes)])

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_frames):
            for i in range(self.n_snakes):    
                self.frames[i].append(obs[i])
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        for i in range(self.n_snakes):
            self.frames[i].append(obs[i])
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        # assert len(self.frames) == self.n_frames
        obs = []
        for i in range(self.n_snakes):
            obs.append(np.concatenate(self.frames[i], axis=2))
        return obs

    def set_foods(self, n):
        self.env.set_foods(n)

    def set_snakes(self, n):
        self.env.set_snakes(n)


INITIAL_HEALTH = 100
class DistanceReward(gym.Wrapper):

    def __init__(self, env):
        super(DistanceReward, self).__init__(env)
        self.health = INITIAL_HEALTH
        self.width = self.env.width
        self.height = self.env.height
        self.snake = self.env.snake
        self.food = self.env.food

    def step(self, action):
        assert self.env.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        prev_snake_head = self.env.snake.head
        snake_tail = self.env.snake.step(action)

        reward = 0.
        done = False
        self.health -= 1
        
        #snake ate food
        if self.env.snake.head == self.env.food:
            reward += 1.
            self.env.snake.snake.append(snake_tail)
            self.health = INITIAL_HEALTH + len(self.env.snake.snake) * 20
            empty_cells = self.env.get_empty_cells()
            self.env.food = empty_cells[self.env.np_random.choice(len(empty_cells))]
        
        #snake collided wall
        elif self.env.is_collided_wall(self.env.snake.head):
            reward -= 1.
            done = True
        
        #snake bite itself 
        elif self.env.snake.head in self.env.snake.body:
            reward -= 1.
            done = True

        else:
            if not self.health:
                done = True
                reward -= 1.
                self.health = INITIAL_HEALTH
            else:
                snake_len = len(self.env.snake.snake)
                prev_distance = self.distance_to_food(prev_snake_head)
                curr_distance = self.distance_to_food(self.env.snake.head)
                if prev_distance < curr_distance:
                    reward -= 0.1
                if prev_distance > curr_distance:
                    reward += 0.1
            
        reward = np.clip(reward, -1., 1.)

        return self.env.get_image(), reward, done, {}

    def distance_to_food(self, head):
        head_x, head_y = head
        food_x, food_y = self.env.food
        return np.sqrt((head_x - food_x)**2 + (head_y - food_y)**2)


