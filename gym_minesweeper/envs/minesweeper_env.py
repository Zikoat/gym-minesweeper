import random
import gym
import numpy as np


class MinesweeperEnv(gym.Env):

    def __init__(self, width=8, height=8, mine_count=10, flood_fill=True,
                 debug=False, punishment=0.01):
        self.width = width
        self.height = height
        self.mines_count = mine_count
        self.debug = debug
        self.flood_fill = flood_fill
        self.punishment = punishment
        self.window = None

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
        prev_game_over = self._game_over()

        self.steps += 1
        x, y = self._parse_action(action)
        self._open_cell(x, y)

        reward = self._get_reward()
        observation = self._get_observation()
        done = self._game_over()

        if self.debug and not prev_game_over and done:
            print("game over")

        return observation, reward, done, self._get_info(prev_game_over, action)

    def reset(self):
        self.observation_space = gym.spaces.Box(low=np.float32(-2),
                                                high=np.float32(8),
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

        if mode == 'human' and not self.window:
            from gym_minesweeper.window import Window
            print("Showing MineSweeper board in own window.\nPyCharm users might want to disable \"Show plots in tool window\".")
            self.window = Window('gym_minigrid')
            self.window.reg_event("button_press_event", self._onclick)
            self.window.show(block=True)

        img = [[COLORS[cell] for cell in row] for row in
               self._get_observation()]

        if mode == 'human':
            self.window.set_caption(
                "reward:" + str(np.round(self._get_reward(), 4)))
            self.window.show_img(img)

        return img

    def _onclick(self, event):
        # secretly flip x and y because we havent cared to do it correctly in the environment
        # todo fix x and y order
        x = round(event.ydata)
        y = round(event.xdata)
        current_action = y * self.width + x
        self.step(current_action)
        self.render(mode="human")

    def close(self):
        if self.window:
            self.window.close()

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
                            and 0 <= iy <= self.height - 1:
                    # self.open_cells[ix, iy] = 1
                        if self._get_neighbor_mines(ix, iy) == 0 and not self.open_cells[ix, iy]:

                            self._open_cell(ix, iy)
                        else:
                            self.open_cells[ix, iy] = 1


    def _get_reward(self):
        openable = self.width * self.height - self.mines_count
        open = np.count_nonzero(self.open_cells)
        open_mines = np.count_nonzero(np.logical_and(self.open_cells, self.mines))
        return (open - self.unnecessary_steps * self.punishment) / openable - open_mines

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

    def _get_info(self, prev_game_over, action):
        return {
            "opened cells": np.count_nonzero(self.open_cells),
            "steps": self.steps,
            "unnecessary steps": self.unnecessary_steps,
            "game over": self._game_over(),
            "died this turn": self._game_over() and not prev_game_over,
            "mine locations": self.mines.astype(int),
            "opened cell": self._parse_action(action)
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
