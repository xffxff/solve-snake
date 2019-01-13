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


class SimpleObservation(gym.Wrapper):

    def __init__(self, env=None):
        super(SimpleObservation, self).__init__(env)

        height = self.env.hight
        width = self.env.width
        low = np.array([0., 0., 0., 0., 0., 1.], np.float32)
        high = np.array([19., 19., 19., 19., 3., height * width], np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        self.action_space = Discrete(4)

    def reset(self):
        self.env.prev_action = SnakeAction.UP
        self.env.snake_head = self.env.snake_start
        self.env.empty_cells = [(x, y) for x in range(self.env.width) for y in range(self.env.hight)]
        self.env.snake.clear()
        self.env.snake.append(self.env.snake_start)
        self.env.empty_cells.remove(self.env.snake_start)
        self.env.food = self.env.empty_cells[self.env.np_random.choice(len(self.env.empty_cells))]
        self.env.empty_cells.remove(self.env.food)
        head_x, head_y = self.env.snake_head
        food_x, food_y = self.env.food
        return np.array([head_x, head_y, food_x, food_y, self.env.prev_action, 1], np.float32)

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
        head_x, head_y = self.env.snake_head
        food_x, food_y = self.env.food
        observation = np.array([head_x, head_y, food_x, food_y, self.env.prev_action, len(self.env.snake)], np.float32)
        return observation, reward, done, {}

    def _distance(self, x, y):
        return np.sum((np.array(x) - np.array(y))**2)

class ComplexObservation(gym.Wrapper):

    def __init__(self, env=None):
        super(ComplexObservation, self).__init__(env)

        height = self.env.hight
        width = self.env.width
        low = np.array([0., 0., 0., 0., 0., 0, 0, 0, 0], np.float32)
        high = np.array([19., 19., 19., 19., 3., 1, 1, 1, 1], np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        self.action_space = Discrete(4)

    def reset(self):
        self.env.prev_action = SnakeAction.UP
        self.env.snake_head = self.env.snake_start
        self.env.empty_cells = [(x, y) for x in range(self.env.width) for y in range(self.env.hight)]
        self.env.snake.clear()
        self.env.snake.append(self.env.snake_start)
        self.env.empty_cells.remove(self.env.snake_start)
        self.env.food = self.env.empty_cells[self.env.np_random.choice(len(self.env.empty_cells))]
        self.env.empty_cells.remove(self.env.food)
        head_x, head_y = self.env.snake_head
        food_x, food_y = self.env.food
        return np.array([head_x, head_y, food_x, food_y, self.env.prev_action, 0, 0, 0, 0], np.float32)

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
        
        left_head, right_head, up_head, down_head = self._is_bite()
        
        head_x, head_y = self.env.snake_head
        food_x, food_y = self.env.food
        observation = np.array([head_x, head_y, food_x, food_y, self.env.prev_action, left_head, right_head, up_head, down_head], np.float32)
        return observation, reward, done, {}

    def _distance(self, x, y):
        return np.sum((np.array(x) - np.array(y))**2)
    
    def _is_bite(self):
        left_bite, right_bite, up_bite, down_bite = 0, 0, 0, 0
        up_head = self.env.next_head(SnakeAction.UP)
        up_head_state = self.env.cell_state(up_head)
        if up_head_state == SnakeCellState.BITE:
            up_bite = 1
        
        down_head = self.env.next_head(SnakeAction.DOWN)
        down_head_state = self.env.cell_state(down_head)
        if down_head_state == SnakeCellState.BITE:
            down_bite = 1

        left_head = self.env.next_head(SnakeAction.LEFT)
        left_head_state = self.env.cell_state(left_head)
        if left_head_state == SnakeCellState.BITE:
            left_bite = 1

        right_head = self.env.next_head(SnakeAction.RIGHT)
        right_head_state = self.env.cell_state(right_head)
        if right_head_state == SnakeCellState.BITE:
            right_bite = 1
        
        return left_bite, right_bite, up_bite, down_bite