"""
JAX-based loss functions for ML models. Each function is curried to store the neural network apply
function and required parameters. The internal functions each take the current neural network parameters,
the input data, and the target labels, and return the specified loss value.
"""


import jax
import jax.numpy as jnp
import optax

import fl.utils.functions


def cross_entropy_loss(net, classes):
    """
    Cross entropy/log loss, best suited for softmax models
    
    Additional arguments:
    - classes: the number of classes in the dataset
    """
    @jax.jit
    def _apply(params, X, y):
        logits = net.apply(params, X)
        labels = jax.nn.one_hot(y, classes)
        return jnp.mean(optax.softmax_cross_entropy(logits, labels))
    return _apply


def ae_l2_loss(net):
    """Autoencoder L2 loss, internal function only takes the neural network parameters and input data"""
    @jax.jit
    def _apply(params, x):
        z = net.apply(params, x)
        return jnp.mean(optax.l2_loss(z, x))
    return _apply


def fedmax_loss(net, net_act, classes):
    """
    Loss function used for the FedMAX algorithm proposed in `https://arxiv.org/abs/2004.03657 <https://arxiv.org/abs/2004.03657>`_
    
    Additional arguments:
    - net_act: the activation function to use for the neural network, this is the output up to the second-last layer
    - classes: the number of classes in the dataset
    """
    @jax.jit
    def _apply(params, X, y):
        logits = net.apply(params, X)
        labels = jax.nn.one_hot(y, classes)
        act = net_act.apply(params, X)
        zero_mat = jnp.zeros(act.shape)
        kld = (lambda x, y: y * (jnp.log(y) - x))(jax.nn.log_softmax(act), jax.nn.softmax(zero_mat))
        return jnp.mean(optax.softmax_cross_entropy(logits, labels)) + jnp.mean(kld)
    return _apply


def smp_loss(net, scale, loss, val_X, val_y, classes):
    """
    Loss function for stealthy model poisoning `https://arxiv.org/abs/1811.12470 <https://arxiv.org/abs/1811.12470>`_,
    assumes a classification task
    
    Additional arguments:
    - scale: the scale of the poisoned loss function over the stealthy loss function
    - val_X: the validation data, used for stealth
    - val_y: the validation labels, used for stealth
    - classes: the number of classes in the dataset
    """
    @jax.jit
    def _apply(params, X, y):
        val_logits = net.apply(params, val_X)
        val_labels = jax.nn.one_hot(val_y, classes)
        return scale * loss(params, X, y) + jnp.mean(optax.softmax_cross_entropy(val_logits, val_labels))
    return _apply


def constrain_distance_loss(alpha, loss, opt, opt_state):
    """
    Loss function from the constrain and scale attack `https://arxiv.org/abs/1807.00459 <https://arxiv.org/abs/1807.00459>`_
    specifically for evading distance metric-based defense systems

    Additional arguments:
    - alpha: weighting of attack loss vs. constraint loss
    - opt: the optimizer to use
    - opt_state: the optimizer state
    """
    @jax.jit
    def _apply(params, X, y):
        global_params = params
        grads = jax.grad(loss)(params, X, y)
        updates, _ = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return alpha * loss(params, X, y) + (1 - alpha) * jnp.mean(optax.l2_loss(fl.utils.functions.tree_flatten(params), fl.utils.functions.tree_flatten(global_params)))
    return _apply


def constrain_cosine_loss(alpha, loss, opt, opt_state):
    """
    Loss function from the constrain and scale attack `https://arxiv.org/abs/1807.00459 <https://arxiv.org/abs/1807.00459>`_
    specifically for evading cosine similarity-based defense systems

    Additional arguments:
    - alpha: weighting of attack loss vs. constraint loss
    - opt: the optimizer to use
    - opt_state: the optimizer state
    """
    @jax.jit
    def _apply(params, X, y):
        global_params = params
        grads = jax.grad(loss)(params, X, y)
        updates, _ = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return alpha * loss(params, X, y) + (1 - alpha) * (
            1 - optax.cosine_similarity(fl.utils.functions.tree_flatten(params),
            fl.utils.functions.tree_flatten(global_params))
        )
    return _apply
