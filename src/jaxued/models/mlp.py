from functools import partial
from typing import Any, NamedTuple, Optional, Sequence, Tuple, Union

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import optax
import pandas as pd
from brax import envs
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
from flax import linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.serialization import from_bytes, msgpack_serialize, to_state_dict
from flax.training import train_state
from flax.training.train_state import TrainState as BaseTrainState
from gymnax.environments import environment, spaces
from gymnax.wrappers.purerl import GymnaxWrapper
from jax import lax
from jax import numpy as jnp
from jax import random
from jax.lib import xla_bridge

'''
Model Classes for feed-forward prediction models
'''
class ForwardMLP(nn.Module):
    density_1: int = 256
    output_dim: int = 5

    def setup(self):
        self.dense1 = nn.Dense(self.density_1)
        self.dense2 = nn.Dense(self.density_1)
        self.dense3 = nn.Dense(self.density_1)
        self.dense4 = nn.Dense(self.density_1)
        self.dense5 = nn.Dense(self.density_1)
        self.dense6 = nn.Dense(self.density_1)
        self.dense7 = nn.Dense(self.density_1)
        self.dense8 = nn.Dense(self.density_1)
        self.dense9 = nn.Dense(self.density_1)
        self.dense10 = nn.Dense(self.density_1)
        self.dense_mean = nn.Dense(self.output_dim)  # Assuming 12-dimensional output mean

    @nn.remat
    def __call__(self, x_batch):
        x = x_batch
        x = self.dense1(x)
        x = nn.relu(x)
        x = self.dense2(x)
        x = nn.relu(x)
        x = self.dense3(x)
        x = nn.relu(x)
        x = self.dense4(x)
        x = nn.relu(x)
        x = self.dense5(x)
        x = nn.relu(x)
        x = self.dense6(x)
        x = nn.relu(x)
        x = self.dense7(x)
        x = nn.relu(x)
        x = self.dense8(x)
        x = nn.relu(x)
        x = self.dense9(x)
        x = nn.relu(x)
        x = self.dense10(x)
        x = nn.relu(x)
        mean = self.dense_mean(x)
        return mean


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)
    
def BCELoss(logits, labels):
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    return -labels * log_p - (1. - labels) * log_not_p

@partial(jax.jit, static_argnums=(3,))
def train_step_MLP(state, x, y, update_gradients=True):
    """Train for a single step."""
    def loss_fn(params):
        pred = state.apply_fn(params, x)
        # loss = jnp.mean(jnp.mean(optax.l2_loss(mean, y),-1),0)
        #the outout is 11 dim obs and 1 dim reward in the end
        loss = jnp.mean(optax.l2_loss(pred, y))
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    if update_gradients: state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def test_step_MLP(state, x, y):
    """Train for a single step."""
    def loss_fn(params):
        pred = state.apply_fn(params, x)
        # loss = jnp.mean(jnp.mean(optax.l2_loss(mean, y),-1),0)
        #the outout is 11-dim obs, 1-dim reward, 1-dim done
        loss = jnp.mean(optax.l2_loss(pred, y))
        return loss
    
    loss = loss_fn(state.params)
    return  loss