
import gym
import numpy as np 
from gym.spaces import Box, Discrete


class SnakeCellState(object):
    EMPTY = 0
    WALL = 1
    BITE = 2
    FOOD = 3

class SnakeReward(object):
    ALIVE = 0.
    FOOD = 1.
    DEAD = -1.
    WON = 100.

class SnakeAction(object):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class RewardDesign(gym.Wrapper):

    def __init__(self, env=None):
        super(RewardDesign, self).__init__(env)
    
    def step(self, action):
        assert self.env.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if not self.env.is_valid_action(action):
            action = self.env.prev_action
        self.env.prev_action = action

        next_head = self.env.next_head(action)
        next_head_state = self.env.cell_state(next_head)

        prev_head = self.env.snake_head

        self.env.snake_head = next_head
        self.env.snake.appendleft(next_head)

        done = False
        if next_head_state == SnakeCellState.WALL:
            reward = SnakeReward.DEAD
            done = True
        elif next_head_state == SnakeCellState.BITE:
            reward = SnakeReward.DEAD
            done = True
        elif next_head_state == SnakeCellState.FOOD:
            if len(self.env.empty_cells) > 0:
                self.env.food = self.env.empty_cells[self.env.np_random.choice(len(self.env.empty_cells))]
                self.env.empty_cells.remove(self.env.food)
                reward = SnakeReward.FOOD
            else:
                reward = SnakeReward.WON
                done = True
        else:
            if self._distance(self.env.snake_head, self.env.food) < self._distance(prev_head, self.env.food):
                reward = 0.1
            else:
                reward = -0.1
            self.env.empty_cells.remove(self.env.snake_head)
            emtpy_cell = self.env.snake.pop()
            self.env.empty_cells.append(emtpy_cell)
        return self.env.get_image(), reward, done, {}
    
    def _distance(self, x, y):
        return np.sum((np.array(x) - np.array(y))**2)


