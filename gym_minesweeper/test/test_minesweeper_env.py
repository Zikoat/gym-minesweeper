from unittest import TestCase, skip
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
        ob, reward, episode_over, info = self.env.step(
            self.env.action_space.sample())

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

    def test_render_ansi(self):
        self.env.step(self.env.action_space.sample())
        print(self.env.render("ansi"))

    def test_render_rgb_array(self):
        self.env.step(self.env.action_space.sample())
        img = self.env.render("rgb_array")
        self.assertEqual(img.shape[2], 3)
        print(img.shape)
        print(img)

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

    def test_seed(self):
        env = gym.make("Minesweeper-v0", seed=0)
        ob, reward, episode_over, info = env.step(1)

        expected_mine_locations = np.array(
            [[0, 0, 1, 0, 1, 0, 1, 1, ],
             [0, 0, 0, 1, 0, 0, 1, 0, ],
             [1, 0, 0, 1, 0, 0, 1, 0, ],
             [0, 0, 0, 0, 0, 0, 0, 0, ],
             [0, 0, 0, 0, 0, 0, 0, 0, ],
             [0, 0, 0, 0, 0, 0, 0, 0, ],
             [0, 0, 0, 0, 0, 0, 0, 0, ],
             [0, 0, 0, 1, 0, 0, 0, 0, ]])

        np.testing.assert_array_equal(
            info["mine locations"],
            expected_mine_locations)

    def test_seeded_reset_changes_locations(self):
        env = gym.make("Minesweeper-v0", seed=0)
        state = env.step(38)
        mine_locations_1 = state[3]["mine locations"]
        env.reset()
        state = env.step(38)
        mine_locations_2 = state[3]["mine locations"]
        self.assertFalse(np.array_equal(mine_locations_1, mine_locations_2))

    def test_first_open_is_mine(self):
        env = gym.make("Minesweeper-v0", seed=0)
        ob, reward, episode_over, info = env.step(31)
        print(info["mine locations"])
        print(env.render("ansi"))
        print(info["opened cell"])
        self.assertTrue(episode_over)

        self.assertEquals(reward, -1)
        self.assertIn(-2, ob)
        self.assertTrue(info["died this turn"])
        self.assertEqual(1, info["steps"])
        self.assertEqual(1, info["opened cells"])

    def test_reset_returns_observation(self):
        observation_reset = self.env.reset()
        observation_step = self.env.step(self.env.action_space.sample())[0]
        expected_observation = np.full((self.env.width, self.env.height), -1)

        self.assertEqual(observation_reset.shape, observation_step.shape)
        self.assertTrue(np.array_equal(expected_observation, observation_reset))

    def test_legal_actions(self):
        prev_legal_actions = self.env.legal_actions()
        self.assertEqual(self.env.legal_actions().size, self.env.action_space.n)
        ob, reward, episode_over, info = self.env.step(
            self.env.action_space.sample())

        current_legal_actions = self.env.legal_actions()
        print(current_legal_actions)
        self.assertLess(current_legal_actions.size, prev_legal_actions.size)

        while not episode_over:
            ob, reward, episode_over, info = self.env.step(
                self.env.legal_actions()[0])

        self.assertTrue(self.env._is_done())
        self.assertEqual(0, info["unnecessary steps"])
