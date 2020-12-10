import logging
from gym.envs import registration

logger = logging.getLogger(__name__)

all_envs = registration.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]
for env in registration.registry.env_specs.copy():
    if 'Minesweeper-v0' in env:
        print("Remove {} from registry".format(env))
        del registration.registry.env_specs[env]
    if 'MinesweeperHard-v0' in env:
        print("Remove {} from registry".format(env))
        del registration.registry.env_specs[env]

registration.register(
    id='Minesweeper-v0',
    entry_point='gym_minesweeper.envs:MinesweeperEnv',
    nondeterministic=False,
)

registration.register(
    id='MinesweeperHard-v0',
    entry_point='gym.envs:MinesweeperHardEnv',
    nondeterministic=False,
)
