import os, subprocess, time, signal
import random
from enum import Enum

import gym
import numpy as np
from gym import error, spaces
from gym import utils
from gym.utils import seeding


class MinesweeperEnv(gym.Env):

    def __init__(self, width=8, height=8, bombs=10, debug=False):
        self.width = width
        self.height = height
        self.bombs = bombs
        self.debug = debug

        self.observation_space = spaces.Box(-2, 8, shape=(width, height))
        self.action_space = spaces.Discrete(width * height)
        self.open_cells = np.zeros((width, height))
        self.mines = self._generate_mines(width, height, bombs)
        print(self.mines)

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        x, y = self._parse_action(action)
        self._open_cell(x, y)
        reward = self._get_reward()
        observation = self._get_observation()
        episode_over = self._game_over()
        if self.debug:
            print("the game is {} over".format("" if episode_over else "not"))
        return observation, reward, episode_over, {}

    def reset(self):
        self.open_cells = np.zeros((self.width, self.height))
        self.mines = self._generate_mines(self.width, self.height, self.bombs)
        return self._get_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def _parse_action(self, action):

        x =  action%self.width
        y =  action//self.height
        print("action is ({},{})".format(x,y))
        return x, y

    def _open_cell(self, x, y):
        self.open_cells[x, y] = 1

    def _get_reward(self):
        pass

    def _generate_mines(self, width, height, bombs):
        mines = np.zeros((width, height))
        mines1d = random.sample(range(width*height), bombs)

        for coord in mines1d:
            x = coord % width
            y = coord // height
            mines[x, y] = 1

        return mines

    def _get_observation(self):
        self.open_cells
        observation = np.zeros(self.open_cells.shape)
        for ix, iy in np.ndindex(self.open_cells.shape):
            open = self.open_cells[ix, iy]
            mine = self.mines[ix, iy]

            if not open:
                observation[ix, iy] = -1
            elif open and mine:
                observation[ix, iy] = -2
            elif open:
                observation[ix, iy] = self._get_neighbor_mines(ix, iy)

            if self.debug:
                print("({},{}) open:{}, mine:{}, observation:{}".format(ix, iy, open, mine, observation[ix, iy]))

        return observation

    def _game_over(self):

        logical_and = np.logical_and(self.open_cells, self.mines)
        if self.debug:
            print("logical and", logical_and)
        return np.any(logical_and)

    def _get_neighbor_mines(self, x, y):
        neighbors = [(x-1,y-1),(x,y-1),(x+1,y-1),(x-1,y),(x+1,y),(x-1,y+1),(x,y+1),(x+1,y+1)]
        mine_count = 0
        for ix, iy  in neighbors:
            if ix>=0 and ix<=self.width and iy>=0 and iy<=self.height:
                mine_count += 1
        return mine_count