from functools import partial

import jax
import numpy as np

"""
General utility library for ymir
"""

def chain(funclist, x):
    """Chain a list of function together and return their composition upon x"""
    for f in funclist:
        x = f(x)
    return x

@partial(jax.jit, static_argnums=(1, 2, 3,))
def tree_uniform(tree, low=0.0, high=1.0, rng=np.random.default_rng()):
    """Create an equivalently shaped tree with random number elements in the range [low, high)"""
    return jax.tree_map(lambda x: rng.uniform(low=low, high=high, size=x.shape), tree)


@partial(jax.jit, static_argnums=(1, 2, 3,))
def tree_add_normal(tree, loc=0.0, scale=1.0, rng=np.random.default_rng()):
    """Add normally distributed noise to each element of the tree, (mu=loc, sigma=scale)"""
    return jax.tree_map(lambda x: x + rng.normal(loc=loc, scale=scale, size=x.shape), tree)


@jax.jit
def tree_mul(tree, scale):
    """Multiply the elements of a pytree by the value of scale"""
    return jax.tree_map(lambda x: x * scale, tree)


@jax.jit
def tree_add(*trees):
    """Element-wise add any number of pytrees"""
    return jax.tree_multimap(lambda *xs: sum(xs), *trees)


@jax.jit
def tree_flatten(tree):
    """Flatten a pytree into a vector"""
    return jax.flatten_util.ravel_pytree(tree)[0]