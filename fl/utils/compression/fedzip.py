"""
The FedZip compression scheme from `https://arxiv.org/abs/2102.01593 <https://arxiv.org/abs/2102.01593>`_
"""

import jax
import jax.numpy as jnp
import numpy as np
from sklearn import cluster


# Endpoint-side FedZip functionality

def encode(all_grads, compress=True):
    """Compress all of the updates, performs a lossy-compression then if compress is True, a lossless compression encoding."""
    return [_encode(g, compress=compress) for g in all_grads]


def _encode(grads, compress=True):
    usable_grads = jax.tree_leaves(jax.tree_map(lambda x: x.flatten(), grads))
    sparse_grads = [_top_z(0.3, np.array(g)) for g in usable_grads]
    quantized_grads = [_k_means(g) for g in sparse_grads]
    if compress:
        encoded_grads = []
        codings = []
        for g in quantized_grads:
            e = _encoding(g)
            encoded_grads.append(e[0])
            codings.append(e[1])
        return encoded_grads, codings
    return jax.tree_multimap(lambda x, y: x.reshape(y.shape), jax.tree_unflatten(jax.tree_structure(grads), quantized_grads), grads)


def _top_z(z, grads):
    z_index = np.ceil(z * grads.shape[0]).astype(np.int32)
    grads[np.argpartition(abs(grads), -z_index)[:-z_index]] = 0
    return grads

def _k_means(grads):
    X = np.array(grads).reshape(-1, 1)
    model = cluster.KMeans(init='random', n_clusters=3 if len(X) >= 3 else len(X), max_iter=4, n_init=1, random_state=0)
    model.fit(X)
    labels = model.predict(grads.reshape((-1, 1)))
    centroids = model.cluster_centers_
    for i, c in enumerate(centroids):
        grads[labels == i] = c[0]
    return grads

def _encoding(grads):
    centroids = jnp.unique(grads).tolist()
    probs = []
    for c in centroids:
        probs.append(((grads == c).sum() / len(grads)).item())
    return _huffman(grads, centroids, probs)

def _huffman(grads, centroids, probs):
    groups = [(p, i) for i, p in enumerate(probs)]
    if len(centroids) > 1:
        while len(groups) > 1:
            groups.sort(key=lambda x: x[0])
            a, b = groups[0:2]
            del groups[0:2]
            groups.append((a[0] + b[0], [a[1], b[1]]))
        groups[0][1].sort(key=lambda x: isinstance(x, list))
        coding = {centroids[k]: v for (k, v) in  _traverse_tree(groups[0][1])}
    else:
        coding = {centroids[0]: 0b0}
    result = jnp.zeros(grads.shape, dtype=jnp.int8)
    for c in centroids:
        result = jnp.where(grads == c, coding[c], result)
    return result, {v: k for k, v in coding.items()}


def _traverse_tree(root, line=0b0):
    if isinstance(root, list):
        return _traverse_tree(root[0], line << 1) + _traverse_tree(root[1], (line << 1) + 0b1)
    return [(root, line)]


# server-side FedZip functionality

class Decode:
    """Update transformation that decodes the input updates."""
    def __init__(self, params, compress=False):
        """
        Construct the encoder.

        Arguments:
        - params: the parameters of the model, used for structure information
        - compress: whether to perform lossless decompression step
        """
        self.params = params
        self.compress = compress

    def __call__(self, all_grads):
        """Get all updates and decode each one."""
        if self.compress:
            return [_huffman_decode(self.params, g, e) for (g, e) in all_grads]
        return all_grads


@jax.jit
def _huffman_decode(params, grads, encodings):
    flat_params, tree_struct = jax.tree_flatten(params)
    final_grads = [jnp.zeros(p.shape, dtype=jnp.float32) for p in flat_params]
    for i, p in enumerate(flat_params):
        for k, v in encodings[i].items():
            final_grads[i] = jnp.where(grads[i].reshape(p.shape) == k, v, final_grads[i])
    return jax.tree_unflatten(tree_struct, final_grads)