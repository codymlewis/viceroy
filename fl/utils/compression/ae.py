"""
Autoencoder compression scheme from `https://arxiv.org/abs/2108.05670 <https://arxiv.org/abs/2108.05670>`_
"""


import jax
import jax.numpy as jnp
import optax
import haiku as hk

from .. import losses


# Autoencoder compression

def _update(opt, loss):
    @jax.jit
    def _apply(params, opt_state, x):
        grads = jax.grad(loss)(params, x)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    return _apply


class Coder:
    """Store the per-endpoint autoencoders and associated variables."""
    def __init__(self, gm_params, num_clients):
        """
        Construct the Coder.

        Arguments:
        - gm_params: the parameters of the global model
        - num_clients: the number of clients connected to the associated controller
        """
        gm_params = jax.flatten_util.ravel_pytree(gm_params)[0]
        param_size = len(gm_params)
        ae = lambda: AE(param_size)
        self.f = hk.without_apply_rng(hk.transform(lambda x: ae()(x)))
        self.fe = hk.without_apply_rng(hk.transform(lambda x: ae().encode(x)))
        self.fd = hk.without_apply_rng(hk.transform(lambda x: ae().decode(x)))
        loss = losses.ae_l2_loss(self.f)
        opt = optax.adam(1e-3)
        self.updater = _update(opt, loss)
        params = self.f.init(jax.random.PRNGKey(0), gm_params)
        self.params = [params for _ in range(num_clients)]
        self.opt_states = [opt.init(params) for _ in range(num_clients)]
        self.datas = [[] for _ in range(num_clients)]
        self.num_clients = num_clients

    def encode(self, grad, i):
        """Encode the updates of the client i."""
        return self.fe.apply(self.params[i], jax.flatten_util.ravel_pytree(grad)[0])

    def decode(self, all_grads):
        """Decode the updates of the clients."""
        return [self.fd.apply(self.params[i], grad) for i, grad in enumerate(all_grads)]

    def add_data(self, grad, i):
        """Add the updates of the client i to the ith dataset."""
        self.datas[i].append(grad)
    
    def update(self, i):
        """Update the ith client's autoencoder."""
        grads = jnp.array(self.datas[i])
        self.params[i], self.opt_states[i] = self.updater(self.params[i], self.opt_states[i], grads)
        self.datas[i] = []


class AE(hk.Module):
    """Autoencoder for compression/decompression"""
    def __init__(self, in_len, name=None):
        """
        Construct the autoencoder, in_len ensures the the output is the same size as the input.
        """
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
    
    def __call__(self, x):
        """Perform the encode and decode steps"""
        x = self.encoder(x)
        return self.decoder(x)
    
    def encode(self, x):
        """Perform just the encode step"""
        return self.encoder(x)

    def decode(self, x):
        """Perform just the decode step"""
        return self.decoder(x)


class Encode:
    """Encoding update transform."""
    def __init__(self, coder):
        """
        Construct the encoder.
        
        Arguments:
        - coder: the autoencoders used for compression
        """
        self.coder = coder

    def __call__(self, all_grads):
        encoded_grads = []
        for i, g in enumerate(all_grads):
            flat_g = jax.flatten_util.ravel_pytree(g)[0]
            self.coder.add_data(flat_g, i)
            self.coder.update(i)
            encoded_grads.append(self.coder.encode(flat_g, i))
        return encoded_grads


class Decode:
    """Decoding update transform."""
    def __init__(self, params, coder):
        """
        Construct the decoder.
        
        Arguments:
        - params: the parameters of the global model, used for structure information
        - coder: the autoencoders used for decompression
        """
        self.unraveller = jax.flatten_util.ravel_pytree(params)[1]
        self.coder = coder

    def __call__(self, all_grads):
        decoded_grads = self.coder.decode(all_grads)
        return [self.unraveller(d) for d in decoded_grads]