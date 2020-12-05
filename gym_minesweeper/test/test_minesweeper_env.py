from unittest import TestCase,skip
import gym
import gym_minesweeper
import numpy as np


class TestMinesweeperEnv(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("Minesweeper-v0")

    def setUp(self) -> None:
        self.env.reset()

    def test_step(self):
        ob, reward, episode_over, info = self.env.step(self.env.action_space.sample())

        self.assertEqual(ob.shape, (8, 8))
        self.assertEqual(ob.shape, self.env.observation_space.shape)
        if episode_over:
            self.assertLess(reward, 0)
            self.assertTrue(info["game over"])
            self.assertTrue(info["died this turn"])
        else:
            self.assertGreater(reward, 0)

        self.assertEqual(info["mine locations"].shape, ob.shape)
        self.assertGreaterEqual(info["opened cells"], 1)
        self.assertEqual(info["unnecessary steps"], 0)
        self.assertEqual(len(info["opened cell"]), len(ob.shape))

    @skip("Opens window")
    def test_render_human(self):
        self.env.step(self.env.action_space.sample())
        self.env.render("human")

    def test_render_terminal(self):
        self.env.step(self.env.action_space.sample())
        self.env.render("terminal")

    def test_render_rgb_array(self):
        self.env.step(self.env.action_space.sample())
        self.assertEqual(np.array(self.env.render("rgb-array")).shape, (8, 8, 3))

    def test_create_probability_matrix_from_solution(self):
        env = gym.make("Minesweeper-v0", width=3, height=2, mine_count=1)
        env.reset()

        # Remove all the mines
        env.mines = np.zeros((env.width, env.height))
        # Plant a mine in the bottom middle cell
        env.mines[1, 1] = 1

        # Open top middle cell
        env.step(1)
        # Open top right cell
        env.step(2)

        self.assertEqual(env.render("ansi"), "x11\nxxx")

