"""
Standard endpoint collaborators for federated learning.
"""

from functools import partial
import jax

class Scout:
    """An endpoint for federated learning, holds its own data and personal learning variables."""
    def __init__(self, opt, opt_state, loss, data, epochs):
        """
        Constructor for a Scout.

        Arguments:
        - opt: optimizer to use for training
        - opt_state: initial optimizer state
        - loss: loss function to use for training
        - data: data to use for training
        - epochs: number of epochs to train for per round
        """
        self.opt_state = opt_state
        self.data = data
        self.batch_size = data.batch_size
        self.epochs = epochs
        self.opt = opt
        self.loss = loss
        self.update = partial(update, opt, loss)


@partial(jax.jit, static_argnums=(0, 1,))
def update(opt, loss, params, opt_state, X, y):
    """
    Local learning step.

    Arguments:
    - opt: optimizer
    - loss: loss function
    - params: model parameters
    - opt_state: optimizer state
    - X: samples
    - y: labels
    """
    grads = jax.grad(loss)(params, X, y)
    updates, opt_state = opt.update(grads, opt_state, params)
    return grads, opt_state, updates