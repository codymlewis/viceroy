"""
Classes and functions for a global model for use within federated learning

Author: Cody Lewis
"""

import torch

from utils.models import load_model
import utils


class GlobalModel:
    """The central global model for use within federated learning"""
    def __init__(self, num_in, num_out, options):
        self.params = options.model_params
        self.params['num_in'] = num_in
        self.params['num_out'] = num_out
        self.net = load_model(self.params)
        self.histories = dict()
        self.fit_fun = load_fit_fun(options.fit_fun)

    def fit(self, grads, params):
        """Fit the model to some client gradients"""
        self.fit_fun(self, grads, params)

    def predict(self, x):
        """Predict the classes of the data x"""
        return self.net(x)

    def get_params(self):
        """Get the tensor form parameters of this model"""
        return self.net.get_params()


def load_fit_fun(fn_name):
    """Load the class of the specified adversary"""
    fit_funs = {
        "federated averaging": fed_avg,
        "foolsgold": foolsgold
    }
    if (chosen_fit_fun := fit_funs.get(fn_name)) is None:
        raise utils.errors.MisconfigurationError(
            f"Fitness function '{fn_name}' does not exist, " +
            f"possible options: {set(fit_funs.keys())}"
        )
    return chosen_fit_fun


def fed_avg(net, grads, _params):
    """Perform federated averaging across the client gradients"""
    with torch.no_grad():
        num_clients = len(grads)
        total_dc = sum([grads[i]["data_count"] for i in range(num_clients)])
        alpha = torch.tensor([0 for _ in range(num_clients)], dtype=torch.float32)
        for i in range(num_clients):
            alpha[i] = grads[i]["data_count"] / total_dc
            if net.net is not None:
                for k, p in enumerate(net.net.parameters()):
                    p.data.add_(alpha[i] * grads[i]["params"][k])
        return alpha


def find_feature_importance(net):
    """Get a vector indicating the importance of features in the network"""
    with torch.no_grad():
        w_t = utils.flatten_params(net.get_params(), net.params)
        return abs(w_t - w_t.mean()) / sum(abs(w_t))


def foolsgold(net, grads, params):
    """Perform FoolsGold learning across the client gradients"""
    with torch.no_grad():
        flat_grads = utils.flatten_grads(grads, net.params)
        num_clients = len(grads)
        cs = torch.tensor(
            [[0 for _ in range(num_clients)] for _ in range(num_clients)],
            dtype=torch.float32
        )
        v = torch.tensor([0 for _ in range(num_clients)], dtype=torch.float32)
        alpha = torch.tensor([0 for _ in range(num_clients)], dtype=torch.float32)
        beta = torch.tensor([1 for _ in range(num_clients)], dtype=torch.float32)
        if len(net.histories) < num_clients:
            while len(net.histories) < num_clients:
                net.histories[len(net.histories)] = flat_grads[len(net.histories)]
        else:
            for i in range(num_clients):
                if params['reputation']:
                    beta[i] = torch.cosine_similarity(net.histories[i], flat_grads[i], dim=0)
                net.histories[i] += flat_grads[i]
        beta = 2 * beta - 1
        if params['importance']:
            feature_importance = find_feature_importance(net)
        else:
            feature_importance = torch.tensor([1]).to(net.params['device'])
        for i in range(num_clients):
            for j in {x for x in range(num_clients)} - {i}:
                cs[i][j] = torch.cosine_similarity(
                    net.histories[i] * feature_importance,
                    net.histories[j] * feature_importance,
                    dim=0
                )
            v[i] = max(cs[i])
        del feature_importance
        for i in range(num_clients):
            for j in range(num_clients):
                if (v[j] > v[i]) and (v[j] != 0):
                    cs[i][j] *= v[i] / v[j]
            alpha[i] = 1 - max(cs[i])
        alpha = alpha / max(alpha)
        ids = alpha != 1
        alpha[ids] = params['kappa'] * (
            torch.log(alpha[ids] / (1 - alpha[ids])) + 0.5)
        alpha[alpha > 1] = 1
        alpha[alpha < 0] = 0
        alpha = alpha / alpha.sum()
        if net.net is not None:
            for k, p in enumerate(net.net.parameters()):
              for i in range(num_clients):
                p.data.add_(
                    alpha[i] *
                    beta[i] *
                    grads[i]['params'][k]
                )
        return alpha
