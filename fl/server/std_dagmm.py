"""
STD-DAGMM algorithm proposed in `https://arxiv.org/abs/1911.12560 <https://arxiv.org/abs/1911.12560>`_ designed to mitigate free-rider attacks
"""

import jax
import jax.flatten_util
import jax.numpy as jnp
import haiku as hk
import optax

import numpy as np
from sklearn import mixture

from . import captain


# Utility functions/classes

class _DA(hk.Module):
    """A simple deep autoencoder model."""
    def __init__(self, in_len, name=None):
        super().__init__(name=name)
        self.encoder = hk.Sequential([
            hk.Linear(60), jax.nn.relu,
            hk.Linear(30), jax.nn.relu,
            hk.Linear(10), jax.nn.relu,
            hk.Linear(1)
        ])
        self.decoder = hk.Sequential([
            hk.Linear(10), jax.nn.tanh,
            hk.Linear(30), jax.nn.tanh,
            hk.Linear(60), jax.nn.tanh,
            hk.Linear(in_len)
        ])

    def __call__(self, X):
        enc = self.encoder(X)
        return enc, self.decoder(enc)


def _loss(net):
    """MSE loss for the deep autoencoder."""
    @jax.jit
    def _apply(params, x):
        _, z = net.apply(params, x)
        return jnp.mean(optax.l2_loss(z, x))
    return _apply


def _da_update(opt, loss):
    """Update function for the autoencoder."""
    @jax.jit
    def _apply(params, opt_state, batch):
        grads = jax.grad(loss)(params, batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    return _apply


@jax.jit
def _relative_euclidean_distance(a, b):
    """Find the relative euclidean distance between two vectors."""
    return jnp.linalg.norm(a - b, ord=2) / jnp.clip(jnp.linalg.norm(a, ord=2), a_min=1e-10)


def _predict(params, net, gmm, X):
    """Make the STD-DAGMM prediction."""
    enc, dec = net.apply(params, X)
    z = jnp.array([[
        jnp.squeeze(e),
        _relative_euclidean_distance(x, d),
        optax.cosine_similarity(x, d),
        jnp.std(x)
    ] for x, e, d in zip(X, enc, dec)])
    return gmm.score_samples(z)


# Algorithm functions/classes


class Captain(captain.ScaleCaptain):
    def __init__(self, params, opt, opt_state, network, rng=np.random.default_rng()):
        super().__init__(params, opt, opt_state, network, rng)
        self.batch_sizes = jnp.array([c.batch_size * c.epochs for c in network.clients])
        x = jnp.array([jax.flatten_util.ravel_pytree(params)[0]])
        self.da = hk.without_apply_rng(hk.transform(lambda x: _DA(x[0].shape[0])(x)))
        rng = jax.random.PRNGKey(42)
        self.da_params = self.da.init(rng, x)
        opt = optax.adamw(0.001, weight_decay=0.0001)
        self.da_opt_state = opt.init(self.da_params)
        self.da_update = _da_update(opt, _loss(self.da))

        self.gmm = mixture.GaussianMixture(4, random_state=0, warm_start=True)

    def update(self, all_grads):
        self.batch_sizes = jnp.array([c.batch_size * c.epochs for c in self.network.clients])
        grads = jnp.array([jax.flatten_util.ravel_pytree(g)[0].tolist() for g in all_grads])
        self.da_params, self.da_opt_state = self.da_update(self.da_params, self.da_opt_state, grads)
        enc, dec = self.da.apply(self.da_params, grads)
        z = jnp.array([[
            jnp.squeeze(e),
            _relative_euclidean_distance(x, d),
            optax.cosine_similarity(x, d),
            jnp.std(x)
        ] for x, e, d in zip(grads, enc, dec)])
        self.gmm = self.gmm.fit(z)

    def scale(self, all_grads):
        grads = jnp.array([jax.flatten_util.ravel_pytree(g)[0].tolist() for g in all_grads])
        energies = _predict(self.da_params, self.da, self.gmm, grads)
        std = jnp.std(energies)
        avg = jnp.mean(energies)
        mask = jnp.where((energies >= avg - std) * (energies <= avg + std), 1, 0)
        total_dc = jnp.sum(self.batch_sizes * mask)
        return (self.batch_sizes / total_dc) * mask