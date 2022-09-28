from abc import ABC, abstractmethod
from typing import Iterable
import numpy as np

import optax
import jax

import fl.utils.functions

from fl.utils.network import Network


class Server(ABC):
    """A FL server, holds the global model and aggregates client updates"""
    params: optax.Params
    network: Network
    opt_state: optax.OptState
    rng: np.random.Generator

    def __init__(self, params, opt, opt_state, network, rng):
        self.params = params
        self.opt_state = opt_state
        self.network = network
        self.rng = rng
        self.update_params = update(opt)

    @abstractmethod
    def update(self, all_grads: Iterable):
        """Update stored values of this object."""
        pass

    @abstractmethod
    def scale(self, all_grads: Iterable) -> jax.Array:
        """Calculate the amount by which to scale the gradients by ($p_i$), according the specified algorithm."""
        pass

    def step(self):
        """
        First get the gradients from the network, use them to update this captain, scale the gradients, then apply the
        a step of the optimizer to the stored global parameters using the sum of those scaled gradients.
        """
        # Client side updates
        all_grads = self.network(self.params, self.rng)

        # Captain side aggregation scaling
        self.update(all_grads)
        alpha = self.scale(all_grads)
        all_grads = apply_scale(alpha, all_grads)

        # Captain side update
        self.params, self.opt_state = self.update_params(self.params, self.opt_state, sum_grads(all_grads))
        return alpha, all_grads


def update(opt):
    """
    Update the global model using endpoint gradients.
    This is a curried function, so first initialize with the selected optimizer.
    The return function may then be used to update the global parameters based on the endpoint gradients
    """
    @jax.jit
    def _apply(params, opt_state, grads):
        updates, opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state
    return _apply


def apply_scale(alpha, all_grads):
    """Scale a collection of gradients by the value of alpha"""
    return [fl.utils.functions.tree_mul(g, a) for g, a in zip(all_grads, alpha)]

def sum_grads(all_grads):
    """Element-wise sum together a collection of gradients, simplifies boilerplate"""
    return fl.utils.functions.tree_add(*all_grads)
