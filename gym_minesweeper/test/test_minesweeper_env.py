from unittest import TestCase
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

        self.assertEqual(np.array(ob).shape, (8, 8))
        if episode_over:
            self.assertEqual(reward, -1)
            self.assertFalse(info["game over"])
            self.assertTrue(info["died this turn"])
            self.assertEqual(info["mine locations"].shape, ob.shape)
        else:
            self.assertGreater(reward, 0)

        self.assertGreaterEqual(info["opened cells"], 1)
        self.assertEqual(info["unnecessary steps"], 0)
        self.assertEqual(len(info["opened cell"]), len(ob.shape))

    def test_render_human(self):
        self.env.render("human")

    def test_render_terminal(self):
        self.skipTest("not implemented")
        self.fail()

    def test_render_rgb_array(self):
        self.skipTest("not implemented")
        self.fail()
