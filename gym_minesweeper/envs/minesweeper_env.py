import random

import cv2
import gym
import numpy as np


class MinesweeperEnv(gym.Env):
    metadata = {'render.modes': ["ansi", "rgb_array", "human"]}
    reward_range = (-float(1), float(1))

    def __init__(self, width=8, height=8, mine_count=10, flood_fill=True,
                 debug=True, punishment=0.01, seed=None):
        self.width = width
        self.height = height
        self.mines_count = mine_count
        self.debug = debug
        self.flood_fill = flood_fill
        self.punishment = punishment

        self.window = None
        self.observation_space = gym.spaces.Box(low=np.float32(-2),
                                                high=np.float32(8),
                                                shape=(self.width, self.height))
        self.action_space = gym.spaces.Discrete(self.width * self.height)
        self.NEIGHBORS = [(-1, -1), (0, -1), (1, -1),
                          (-1, 0), (1, 0),
                          (-1, 1), (0, 1), (1, 1)]
        self.open_cells = np.zeros((self.width, self.height))
        random.seed(a=seed)
        self.mines = self._generate_mines(self.width, self.height,
                                          self.mines_count)
        self.steps = 0
        self.unnecessary_steps = 0

        if self.debug:
            self._assert_invariants()

    def step(self, action):
        """

        Parameters
        ----------
        action (int) :
            A z-order zero-based index indicating which cell to dig.
            If the board is of width 5 and height 2, an action of 0 means the
            upper left corner, 4 means the upper right corner, 5 is the lower
            left corner and 9 is the lower right corner.

            The actual cells that change are determined by standard minesweeper
            rules, eg. if you open a cell with 0 mines around it, a larger space
            will be opened.

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (np.ndarray) :
                An int array of shape (width, height) with integer values
                ranging from -2 to 8 where
                -2 = opened mine
                -1 = closed cell
                0-8 = Amount of mines in the surrounding cells
            reward (float) :
                A value between -1 and 1. Increases from 0 to 1 while the game
                is played based on how many cells have been opened.
                Pressing on a mine reduces the reward to the range -1 to 0.
                Playing a perfect game gives a reward of 1.
                Pressing on an already-opened cell decreases the score by
                self.punishment (default 0.01).
            episode_over (bool) :
                If a mine has been pressed. If reset() is not called, you could
                theoretically continue playing, but this is not advised.
            info (dict) :
                 diagnostic information useful for debugging.
                 Includes:
                    opened cells: Amount of opened cells. Affects reward.
                    steps: The amount of steps taken in this episode.
                    unnecessary steps: The amount of steps that had no effect.
                    game over: If a mine has been opened
                    died this turn: If a mine has been opened this turn.
                    mine locations: The location of all the mines.
                    opened cell: The (x, y) coordinates of the cell that was
                        opened this step.
                 Official evaluations of your agent are not allowed to
                 use this for learning.
        """
        prev_game_over = self._game_over()

        self.steps += 1
        x, y = self._parse_action(action)
        self._open_cell(x, y)

        reward = self._get_reward()
        observation = self._get_observation()
        done = self._is_done()

        if self.debug and not prev_game_over and done:
            print("game over")

        if self.debug:
            self._assert_invariants()

        return observation, reward, done, self._get_info(prev_game_over, action)

    def reset(self):
        self.open_cells = np.zeros((self.width, self.height))
        self.mines = self._generate_mines(self.width, self.height,
                                          self.mines_count)
        self.steps = 0
        self.unnecessary_steps = 0
        return self._get_observation()

    def render(self, mode='human'):
        if mode == "ansi":
            row_strings = []
            for row in self._get_observation().T:
                row_string = ""
                for cell in row:
                    if cell == -1:
                        character = "x"
                    elif cell == 0:
                        character = "."
                    elif cell == -2:
                        character = "B"
                    else:
                        character = str(int(cell))

                    row_string += character
                row_strings.append(row_string)
            return "\n".join(row_strings)

        elif mode == 'human' and not self.window:
            from gym_minesweeper.window import Window
            print("Showing MineSweeper board in own window.\nPyCharm users might want to disable \"Show plots in tool window\".")
            self.window = Window('gym_minigrid')
            self.window.reg_event("button_press_event", self._onclick)
            self.window.show(block=True)

        elif mode == 'human':
            img = [[COLORS[cell] for cell in row] for row in
                   self._get_observation()]

            self.window.set_caption(
                "reward:" + str(np.round(self._get_reward(), 4)))
            self.window.show_img(img)

        elif mode == "rgb_array":
            img = [[COLORS[cell] for cell in row] for row in
                   self._get_observation()]
            img = np.array(img, dtype=np.uint8)
            zoom = 20
            img = cv2.resize(img, dsize=(0,0), fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
            return img
        else:
            print("Did not understand rendering mode. Use any of mode=", self.metadata["render.modes"])

    def _onclick(self, event):
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
        y = action // self.width
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

    def _generate_mines(self, width, height, mine_count):
        mines = np.zeros((width, height))
        print("using seed")
        mines1d = random.sample(range(width * height), mine_count)

        for coord in mines1d:
            x = coord % width
            y = coord // width
            try:
                mines[x, y] = 1
            except:
                raise

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

    def _assert_invariants(self):
        assert self._get_observation().shape == self.observation_space.shape

        if self._game_over():
            assert -1 <= self._get_reward() < 0, \
                "Game is over, but score is {}".format(self._get_reward())
            assert np.count_nonzero(np.logical_and(self.open_cells, self.mines)) == 1, \
                "Game is over, but opened cells is {}".format(np.count_nonzero(np.logical_and(self.open_cells, self.mines)))
        else:
            assert 0 <= self._get_reward() < 1, \
                "Game is not over, but score is {}".format(self._get_reward())
            assert np.count_nonzero(np.logical_and(self.open_cells, self.mines)) == 0, \
                "Game is not over, but opened mines: {}".format(np.count_nonzero(np.logical_and(self.open_cells, self.mines)))

        assert (np.count_nonzero(self.open_cells) == 1 and self._game_over()) \
               == (self._get_reward() == -1), \
            "Game over: {}, mines opened: {}, but score is {}".format(self._game_over(), np.count_nonzero(self.open_cells), self._get_reward())

        assert (np.count_nonzero(self.open_cells) == self.width*self.height) \
               == (self._get_reward() == 1), \
            "The game is won ({}), and the score should be 1, but the score is {}".format(np.count_nonzero(self.open_cells) == self.width*self.height, self._get_reward())

        assert (np.count_nonzero(self.open_cells) == 0) \
               == (self._get_reward() == 0), \
            "The game has just started, but the reward is not zero. reward:{}".format(self._get_reward())

    def _is_done(self):
        openable = self.width * self.height - self.mines_count
        opened = np.count_nonzero(self.open_cells)
        all_opened = opened == openable
        return self._game_over() or all_opened


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
