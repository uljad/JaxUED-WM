from typing import Any, NamedTuple, Sequence

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.wrappers.purerl import GymnaxWrapper
from gymnax.environments import environment
from jax import lax, random

from flax import struct

@struct.dataclass
class SwitchParamsEnvState:
    params: Any
    env_state: environment.EnvState

class SwitchParams(GymnaxWrapper):
    def __init__(self, env, params, num_world_models):
        super().__init__(env)
        self.params = params # SHAPE (1000, ...weights)
        self.num_world_models = num_world_models

    def reset(self, key, params=None):
        index = jax.random.randint(key, (), 0, self.num_world_models)
        params_to_use = jax.tree_util.tree_map(lambda x: x[index], self.params)
        obs, env_state = self._env.reset(key, params)
        return obs, SwitchParamsEnvState(params_to_use, env_state)

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params, state.params)
        state = state.replace(env_state=env_state)
        return obs, state, reward, done, info