import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Autonomous driving
# --------------------------------------------------

register(
    id='DeepCars-v0',
    entry_point='gym_deepcars.envs:DeepCarsEnv'
        )

register(
    id='DeepCars-v1',
    entry_point='gym_deepcars.envs:DeepCarsEnv_v1'
        )

register(
    id='DeepCars-v2',
    entry_point='gym_deepcars.envs:DeepCarsEnv_v2'
        )

register(
    id='DeepCars-v3',
    entry_point='gym_deepcars.envs:DeepCarsEnv_v3'
        )
