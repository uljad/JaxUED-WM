import os
import pickle

from chex._src.pytypes import PRNGKey
import jax
import jax.numpy as jnp
from jax import lax
from typing import Callable, Tuple, Optional, Any
import chex
from flax import struct
from gymnax.environments import spaces
from gymnax.wrappers.purerl import GymnaxWrapper

from jaxued.environments.underspecified_env import UnderspecifiedEnv

from jaxued.environments.world_models.util import SwitchParamsEnvState
from jaxued.environments.world_models.gymnax.ParametricCartpole import ParametricCartpole, EnvParams, EnvState

from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper

@struct.dataclass
class Level:
    index: int

class CartPoleWM(UnderspecifiedEnv):
    def __init__(self, model_paths):
        num_world_models = len(model_paths)
        print(f"We have {num_world_models} world models")
        
        with open(model_paths[0], 'rb') as f: 
            all_params = pickle.load(f)
            all_params = jax.tree_util.tree_map(lambda x: x[None], all_params)

        for filej in model_paths[1:]:
            with open(filej, 'rb') as f:
                params_new = pickle.load(f)
            all_params = jax.tree_util.tree_map(lambda all, new: jnp.concatenate([all, new[None]], axis=0), all_params, params_new)

        env = ParametricCartpole()
        # env_params = env.default_params
        # env = FlattenObservationWrapper(env)
        # env = LogWrapper(env)

        self._env = env
        self.params = all_params
        self.num_world_models = num_world_models

    def step_env(self, rng: jax.Array, state: EnvState, action: int | float, params: EnvParams) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(rng, state.env_state, action, params, state.params)
        state = state.replace(env_state=env_state)
        return lax.stop_gradient(obs), lax.stop_gradient(state), reward, done, info
    
    def reset_env_to_level(self, rng: PRNGKey, level: Level, params: EnvParams) -> Tuple[Any | EnvState]:
        index = level.index
        params_to_use = jax.tree_util.tree_map(lambda x: x[index], self.params)
        obs, env_state = self._env.reset(rng, params)

        return obs, SwitchParamsEnvState(params_to_use, env_state)

    @property
    def default_params(self) -> EnvParams:
        return self._env.default_params

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return self._env.action_space(params)

def make_eval_levels_and_names():
    length     = jnp.logspace(jnp.log10(0.05), jnp.log10(10), num=10)
    masspole   = jnp.logspace(jnp.log10(0.05), jnp.log10(10), num=10)

    def get_arr(length, mass):
        return jnp.array([length, mass])
    
    def make_level(v):
        length, mass = v
        return Level(length=length, masspole=mass, total_mass=1.0 + mass, polemass_length=length * mass)
    

    arrs = jax.vmap(jax.vmap(get_arr, (0, None)), (None, 0))(length, masspole).reshape(-1, 2)

    levels = jax.vmap(make_level)(arrs)
    default = Level()
    levels = jax.tree_map(lambda x, new: jnp.concatenate([x, jnp.array(new)[None]], axis=0), levels, default)
    return levels, [f"length_{i:<2}_mass_{j:<2}" for i, j in arrs] + ['default']

def make_level_generator(num_world_models) -> Callable[[chex.PRNGKey], Level]:
    def sample(rng: chex.PRNGKey) -> Level:
        index = jax.random.randint(rng, (1,), 0, num_world_models)[0]
        return Level(
            index = index,
        ) # default
    return sample
