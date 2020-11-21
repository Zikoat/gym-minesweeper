import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Minesweeper-v0',
    entry_point='gym_minesweeper.envs:MinesweeperEnv',
    nondeterministic=False,  # todo: figure out what this setting does
)

register(
    id='MinesweeperMedium-v0',
    entry_point='gym_soccer.envs:MinesweeperMediumEnv',
    nondeterministic=False,
)

register(
    id='MinesweeperHard-v0',
    entry_point='gym.envs:MinesweeperHardEnv',
    nondeterministic=False,
)
