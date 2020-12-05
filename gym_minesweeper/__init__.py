import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Minesweeper-v0',
    entry_point='gym_minesweeper.envs:MinesweeperEnv',
    nondeterministic=True,  # todo seeding
)

register(
    id='MinesweeperHard-v0',
    entry_point='gym.envs:MinesweeperHardEnv',
    nondeterministic=True,
)
