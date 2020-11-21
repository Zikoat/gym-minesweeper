import random

import gym
import numpy as np
from gym_minigrid.window import Window

class MinesweeperEnv(gym.Env):

    def __init__(self, width=8, height=8, mine_count=10, flood_fill=True,
                 debug=False, punishment=0):
        self.width = width
        self.height = height
        self.mines_count = mine_count
        self.debug = debug
        self.flood_fill = flood_fill
        self.punishment = punishment
        self.window = Window('gym_minesweeper')
        self.window.reg_event("button_press_event", self._onclick)

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
        self.steps += 1
        x, y = self._parse_action(action)
        self._open_cell(x, y)
        reward = self._get_reward()
        observation = self._get_observation()
        episode_over = self._game_over()
        if self.debug:
            print("the game is {} over".format("" if episode_over else "not"))
        return observation, reward, episode_over, self._get_info()

    def reset(self):
        self.observation_space = gym.spaces.Box(-2, 8,
                                            shape=(self.width, self.height))
        self.action_space = gym.spaces.Discrete(self.width * self.height)

        self.open_cells = np.zeros((self.width, self.height))
        self.mines = self._generate_mines(self.width, self.height,
                                          self.mines_count)
        self.steps = 0
        self.unnecessary_steps = 0
        self.NEIGHBORS = [(-1, -1), (0, -1), (1, -1),
                          (-1, 0), (1, 0),
                          (-1, 1), (0, 1), (1, 1)]
        return self._get_observation()

    def render(self, mode='human'):
        if mode == "terminal":
            print()
            for row in self._get_observation():
                rowstring = ""
                for cell in row:
                    if cell == -1:
                        character = "."
                    elif cell == 0:
                        character = " "
                    elif cell == -2:
                        character = "X"
                    else:
                        character = str(int(cell))

                    rowstring += character + " "
                print(rowstring)
            print()

        if mode == "rgb_array":
            return [[COLORS[cell] for cell in row] for row in self._get_observation()]

        if mode == "human":
            img = self.render(mode="rgb_array")
            self.window.show_img(img)

    def _onclick(self, event):
        # secretly flip x and y because we havent cared to do it correctly in the environment
        # todo fix x and y order
        x = round(event.ydata)
        y = round(event.xdata)
        current_action = y * self.width + x
        self.step(current_action)
        self.render(mode="human")


    def close(self):
        pass

    def _parse_action(self, action):
        x = action % self.width
        y = action // self.height
        return x, y

    def _open_cell(self, x, y):
        if self.open_cells[x, y]:
            self.unnecessary_steps += 1
        else:
            if self.debug:
                print("opening cell ({},{})".format(x, y))
            self.open_cells[x, y] = 1
            if self._get_neighbor_mines(x, y) == 0 and self.flood_fill:
                for dx, dy in self.NEIGHBORS:
                    ix, iy = (dx + x, dy + y)
                    if 0 <= ix <= self.width - 1 \
                            and 0 <= iy <= self.height - 1 \
                            and not self.open_cells[ix, iy]:
                        # self.open_cells[ix, iy] = 1
                        if self._get_neighbor_mines(ix, iy) == 0:
                            self._open_cell(ix, iy)

    def _get_reward(self):
        openable = self.width * self.height - self.mines_count
        open = np.count_nonzero(self.open_cells)
        return open / openable - \
               (
                       self.steps - self.unnecessary_steps) * self.punishment / openable

    def _generate_mines(self, width, height, bombs):
        mines = np.zeros((width, height))
        mines1d = random.sample(range(width * height), bombs)

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

        return observation

    def _game_over(self):
        logical_and = np.logical_and(self.open_cells, self.mines)
        return np.any(logical_and)

    def _get_neighbor_mines(self, x, y):
        mine_count = 0
        for dx, dy in self.NEIGHBORS:
            ix, iy = (dx + x, dy + y)
            if 0 <= ix <= self.width - 1 and \
                    0 <= iy <= self.height - 1 and \
                    self.mines[ix, iy]:
                mine_count += 1
        return mine_count

    def _get_info(self):
        return {
            "opened cells": np.count_nonzero(self.open_cells),
            "steps": self.steps,
            "unnecessary steps": self.unnecessary_steps
        }


COLORS = {
    -2: [255, 0, 255],
    -1: [128, 128, 128],
    0: [255, 255, 255],
    1: [0, 0, 255],
    2: [0, 128, 0],
    3: [255, 0, 0],
    4: [0, 0, 128],
    5: [128, 0, 0],
    6: [0, 128, 128],
    7: [255, 255, 0],
    8: [255, 0, 255]
}
