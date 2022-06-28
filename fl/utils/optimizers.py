"""
Unique optimizers proposed in the FL literature
"""


from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
import chex


def pgd(learning_rate, mu, local_epochs=1):
    """
    Perturbed gradient descent proposed as the mechanism for FedProx in `https://arxiv.org/abs/1812.06127 <https://arxiv.org/abs/1812.06127>`_
    """
    return optax.chain(
        _add_prox(mu, local_epochs),
        optax.scale(learning_rate)
    )


class PgdState(NamedTuple):
    """Perturbed gradient descent optimizer state"""
    params: optax.Params
    """Model parameters from most recent round."""
    counter: chex.Array
    """Counter for the number of epochs, determines when to update params."""


def _add_prox(mu: float, local_epochs: int) -> optax.GradientTransformation:
    """
    Adds a regularization term to the optimizer.
    """

    def init_fn(params: optax.Params) -> PgdState:
        return PgdState(params, jnp.array(0))

    def update_fn(grads: optax.Updates, state: PgdState, params: optax.Params) -> tuple:
        if params is None:
            raise ValueError("params argument required for this transform")
        updates = jax.tree_multimap(lambda g, w, wt: g + mu * ((w - g) - wt), grads, params, state.params)
        return updates, PgdState(
            jax.lax.cond(state.counter == 0, lambda _: params, lambda _: state.params, None), (state.counter + 1) % local_epochs
        )

    return optax.GradientTransformation(init_fn, update_fn)



def smp_opt(opt, rho):
    """Optimizer for stealthy model poisoning https://arxiv.org/abs/1811.12470"""
    return optax.chain(
        _add_stealth(rho),
        opt
    )


def _add_stealth(rho: float) -> optax.GradientTransformation:
    """
    Adds a stealth regularization term to the optimizer.
    """

    def init_fn(params: optax.Params) -> None:
        return None

    def update_fn(grads: optax.Updates, state: optax.OptState, params: optax.Params) -> tuple:
        if params is None:
            raise ValueError("params argument required for this transform")
        updates = jax.tree_multimap(lambda g, w: g + rho * jnp.linalg.norm((w - g) - w, ord=2), grads, params)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)