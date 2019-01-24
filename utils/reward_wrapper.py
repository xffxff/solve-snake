
import gym
from gym.wrappers import TimeLimit
import numpy as np


class DistanceReward(gym.Wrapper):

    def __init__(self, env):
        super(DistanceReward, self).__init__(env)

    def step(self, action):
        assert self.env.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if not self.env.is_valid_action(action):
            action = self.env.prev_act
        self.env.prev_act = action

        prev_snake_head = self.env.snake.head
        snake_tail = self.env.snake.step(action)

        reward = 0.
        done = False
        
        #snake ate food
        if self.env.snake.head == self.env.food:
            reward += 1.
            self.env.snake.snake.append(snake_tail)
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
