import functools
from typing import Any, Dict, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from gymnax.environments import environment, spaces
from jax import lax, random

import gymnax
from jaxued.models.mlp import ForwardMLP


@struct.dataclass
class EnvState(environment.EnvState):
    x: jnp.ndarray
    x_dot: jnp.ndarray
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    time: int

@struct.dataclass
class EnvParams(environment.EnvParams):
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    total_mass: float = 1.0 + 0.1  # (masscart + masspole)
    length: float = 0.5
    polemass_length: float = 0.05  # (masspole * length)
    force_mag: float = 10.0
    tau: float = 0.02
    theta_threshold_radians: float = 12 * 2 * jnp.pi / 360
    x_threshold: float = 2.4
    max_steps_in_episode: int = 500  # v0 had only 200 steps!


class ParametricCartpole(environment.Environment):

    def __init__(self):
        super().__init__()
        self.obs_shape = (4,)
        self.forward_model = ForwardMLP()
        self.oracle_env, self.oracle_env_params = gymnax.make("CartPole-v1")
        self.oracle_state = self.oracle_env.reset(random.PRNGKey(0), self.oracle_env_params)[1]
        self.init_model_params = self.forward_model.init(random.PRNGKey(100), jnp.ones((1,5)))

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for CartPole-v1
        return EnvParams()
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: Optional[EnvParams],
        all_params,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(key, state, action, params, all_params)
        obs_re, state_re = self.reset_env(key_reset, params)
        state = jax.tree_map(
            lambda x, y: jax.lax.select(jnp.squeeze(done), x, y), state_re, state_st
        )
        obs = jax.lax.select(jnp.squeeze(done), obs_re, obs_st)
        return obs, state, reward, done, info

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
        all_params,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        
        # all_params = self.init_model_params
        # forward model for observation and reward
        obs = self.get_obs(state)
        action_ = jnp.array([action])
        obs_ac = jnp.concatenate((action_, obs))
        obs_ac = jnp.expand_dims(obs_ac, axis=0)
        mean = self.forward_model.apply(all_params, obs_ac)
        mean = jnp.squeeze(mean)
        obs = mean[:-1]
        reward = mean[-1]
        
        #grounding
        current_time = state.time
        rng, rng_ = jax.random.split(key)
        oracle_obs, new_oracle_state, reward_oracle, oracle_done, _ = self.oracle_env.step(rng, state, action, self.oracle_env_params)
        
        ##ground obs every step
        # state = EnvState(x = oracle_obs[0], 
        #                  x_dot = oracle_obs[1], 
        #                  theta = oracle_obs[2], 
        #                  theta_dot = oracle_obs[3], 
        #                  time = current_time + 1)
        
        #keep rolling observations
        state = EnvState(x = obs[0], 
                         x_dot = obs[1], 
                         theta = obs[2], 
                         theta_dot = obs[3], 
                         time = current_time + 1)
        
        info = {"discount": self.discount(state, params)}

        done = oracle_done
        # done = self.is_terminal(state, params)

        return obs, state, reward, done, info
    
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        init_state = jax.random.uniform(key, minval=-0.05, maxval=0.05, shape=(4,))

        state = EnvState(
            x=init_state[0],
            x_dot=init_state[1],
            theta=init_state[2],
            theta_dot=init_state[3],
            time=0,
        )

        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([state.x, state.x_dot, state.theta, state.theta_dot])

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Check termination criteria
        x = state.x
        theta = state.theta
        done1 = jnp.logical_or(
            x < -params.x_threshold,
            x > params.x_threshold,
        )
        done2 = jnp.logical_or(
            theta < -params.theta_threshold_radians,
            theta > params.theta_threshold_radians,
        )

        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        done = jnp.logical_or(jnp.logical_or(done1, done2), done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "CartPole-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array(
            [
                params.x_threshold * 2,
                jnp.finfo(jnp.float32).max,
                params.theta_threshold_radians * 2,
                jnp.finfo(jnp.float32).max,
            ]
        )
        return spaces.Box(-high, high, (4,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        high = jnp.array(
            [
                params.x_threshold * 2,
                jnp.finfo(jnp.float32).max,
                params.theta_threshold_radians * 2,
                jnp.finfo(jnp.float32).max,
            ]
        )
        return spaces.Dict(
            {
                "x": spaces.Box(-high[0], high[0], (), jnp.float32),
                "x_dot": spaces.Box(-high[1], high[1], (), jnp.float32),
                "theta": spaces.Box(-high[2], high[2], (), jnp.float32),
                "theta_dot": spaces.Box(-high[3], high[3], (), jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )