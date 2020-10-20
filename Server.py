"""
Classes and functions for the server networking aspect of federated learning

Author: Cody Lewis
"""

import torch.nn as nn

import GlobalModel
import Client
import utils


class Server:
    """Federated learning server class"""
    def __init__(self, num_in, num_out):
        self.net = GlobalModel.GlobalModel(num_in, num_out)
        self.num_clients = 0
        self.clients = []

    def fit(self, X, Y, epochs):
        criterion = nn.CrossEntropyLoss()
        for e in range(epochs):
            grads = dict()
            losses = dict()
            for i, c in enumerate(self.clients):
                c.net.copy_params(self.net.get_params())
                losses[i], grads[i] = c.fit(verbose=False)
            self.net.fit(1, grads)
            print(
                f"Epoch: {e + 1}/{epochs}, Loss: {criterion(self.net.predict(X), Y[0])}",
                end="\r"
            )
        print()

    def add_clients(self, clients):
        """Add clients to the server"""
        self.num_clients += len(clients)
        self.net.add_client()
        self.clients.extend(clients)


def fit_client(client):
    return client.fit(verbose=True)


if __name__ == '__main__':
    print("Loading Datasets and Creating Models...")
    train_data = utils.load_data("mnist", train=True)
    val_data = utils.load_data("mnist", train=False)
    server = Server(val_data['x_dim'], val_data['y_dim'])
    server.add_clients([Client.Client(train_data) for _ in range(5)])
    print("Done.")
    print("Starting training...")
    server.fit(val_data['x'], val_data['y'], 2)
