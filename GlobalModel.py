"""
Classes and functions for a global model for use within federated learning

Author: Cody Lewis
"""

import torch
import torch.nn as nn

import SoftMaxModel


class GlobalModel:
    """The central global model for use within federated learning"""
    def __init__(self, num_in, num_out):
        self.net = SoftMaxModel.SoftMaxModel(num_in, num_out)
        self.histories = []
        self.num_clients = 0

    def fit(self, kappa, grads):
        """Fit the model to some client gradients"""
        fed_avg(self.net, self.num_clients, grads, self.net.lr)
        # foolsgold(self.histories, self.num_clients, kappa, grads)

    def predict(self, x):
        """Predict the classes of the data x"""
        return self.net(x)

    def add_client(self):
        """Add a client to the federated learning system"""
        self.num_clients += 1
        self.histories.append(torch.tensor([0]))

    def get_params(self):
        """Get the tensor form parameters of this model"""
        return [p.data for p in self.net.parameters()]


def fed_avg(net, num_clients, grads, lr):
    """Perform federated averaging across the client gradients"""
    total_dc = sum([grads[i]["data_count"] for i in range(num_clients)])
    for k, p in enumerate(net.parameters()):
        for i in range(num_clients):
            with torch.no_grad():
                p.data.sub_(
                    lr *
                    (grads[i]["data_count"] / total_dc) *
                    grads[i]["params"][k]
                )


def foolsgold(histories, num_clients, kappa, grads):
    """Perform FoolsGold learning across the client gradients"""
    # Maybe have a flat grads and list grads
    cs = torch.tensor(
        [[0 for _ in num_clients] for _ in num_clients]
    )
    v = torch.tensor([0 for _ in num_clients])
    alpha = torch.tensor([0 for _ in num_clients])
    for i in range(num_clients):
        histories[i] += grads[i]
        # TODO: feature importances S_t
        for j in {x for x in range(num_clients)} - {i}:
            cs[i][j] = torch.cosine_similarity(
                histories[i],
                histories[j],
                dim=0
            )
        v[i] = max(cs[i])
    for i in range(num_clients):
        for j in range(num_clients):
            if (v[j] > v[i]) and (v[j] != 0):
                cs[i][j] *= v[i] / v[j]
        alpha[i] = 1 - max(cs[i])
    alpha = alpha / max(alpha)
    alpha = kappa * (torch.log(alpha / (1 - alpha)) + 0.5)
    # for k, p in enumerate(net.parameters()):
    #   for i in range(num_clients):
    #       p.data.add_(alpha[i] * grads[i][k])


if __name__ == '__main__':
    x = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=torch.float32)
    y = torch.tensor([
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1]
    ], dtype=torch.float32)
    server = GlobalModel(len(x[0]), len(y[0]))
    client = SoftMaxModel.SoftMaxModel(len(x[0]), len(y[0]))
    client.copy_params(server.get_params())
    server.add_client()
    epochs = 5000
    criterion = nn.BCELoss()
    for i in range(epochs):
        history, grads = client.fit(x, y, 1, verbose=False)
        grads = {0: grads}
        server.fit(1, grads)
        print(
            f"Epoch {i + 1}/{epochs}, client loss: {history['loss'][-1]}, server loss {criterion(server.predict(x), y)}",
            end="\r"
        )
    print()
    print()
    print(f"Client's final predictions:\n{client(x)}")
    print()
    print(f"Server's final predictions:\n{server.predict(x)}")
