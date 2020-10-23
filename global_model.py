"""
Classes and functions for a global model for use within federated learning

Author: Cody Lewis
"""

import torch

from softmax_model import SoftMaxModel


class GlobalModel:
    """The central global model for use within federated learning"""
    def __init__(self, num_in, num_out, fit_fun_name):
        self.net = SoftMaxModel(num_in, num_out)
        self.histories = []
        self.fit_fun = {
            "fed_avg": fed_avg,
            "foolsgold": foolsgold
        }[fit_fun_name]

    def fit(self, grads, params):
        """Fit the model to some client gradients"""
        self.fit_fun(self.net, grads, params)

    def predict(self, x):
        """Predict the classes of the data x"""
        return self.net(x)

    def add_client(self):
        """Add a client to the federated learning system"""
        self.histories.append(torch.tensor([0]))

    def get_params(self):
        """Get the tensor form parameters of this model"""
        return self.net.get_params()


def fed_avg(net, grads, params):
    """Perform federated averaging across the client gradients"""
    num_clients = len(grads)
    total_dc = sum([grads[i]["data_count"] for i in range(num_clients)])
    for k, p in enumerate(net.parameters()):
        for i in range(num_clients):
            with torch.no_grad():
                p.data.sub_(
                    params['lr'] *
                    (grads[i]["data_count"] / total_dc) *
                    grads[i]["params"][k]
                )


def foolsgold(net, grads, params):
    """Perform FoolsGold learning across the client gradients"""
    # Maybe have a flat grads and list grads
    num_clients = len(grads)
    cs = torch.tensor(
        [[0 for _ in num_clients] for _ in num_clients]
    )
    v = torch.tensor([0 for _ in num_clients])
    alpha = torch.tensor([0 for _ in num_clients])
    for i in range(num_clients):
        net.histories[i] += grads[i]
        # TODO: feature importances S_t
        for j in {x for x in range(num_clients)} - {i}:
            cs[i][j] = torch.cosine_similarity(
                net.histories[i],
                net.histories[j],
                dim=0
            )
        v[i] = max(cs[i])
    for i in range(num_clients):
        for j in range(num_clients):
            if (v[j] > v[i]) and (v[j] != 0):
                cs[i][j] *= v[i] / v[j]
        alpha[i] = 1 - max(cs[i])
    alpha = alpha / max(alpha)
    alpha = params['kappa'] * (torch.log(alpha / (1 - alpha)) + 0.5)
    # for k, p in enumerate(net.parameters()):
    #   for i in range(num_clients):
    #       p.data.add_(alpha[i] * grads[i][k])
