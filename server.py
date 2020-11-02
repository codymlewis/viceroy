"""
Classes and functions for the server networking aspect of federated learning

Author: Cody Lewis
"""

import threading

import torch.nn as nn

from global_model import GlobalModel
import utils


class Server:
    """Federated learning server class"""
    def __init__(self, num_in, num_out, options):
        self.net = GlobalModel(num_in, num_out, options.fit_fun)
        self.num_clients = 0
        self.clients = []
        self.options = options

    def fit(self, X, Y, epochs):
        criterion = nn.CrossEntropyLoss()
        for e in range(epochs):
            grads = []
            losses = []
            for c in self.clients:
                c.net.copy_params(self.net.get_params())
                l, g = c.fit()
                grads.append(g)
                losses.append(l)
            self.net.fit(grads, self.options.params)
            stats = utils.find_stats(self.net, X, Y, self.options)
            if self.options.verbosity > 1:
                print(f"Logging stats to {self.options.result_log_file}...")
            utils.log_stats(self.options.result_log_file, stats)
            if self.options.verbosity > 1:
                print("Done logging.")
            if self.options.verbosity > 0:
                print(
                    f"Epoch: {e + 1}/{epochs}, " +
                    f"Loss: {criterion(self.net.predict(X), Y[0]):.6f}, " +
                    f"Accuracy: {stats['accuracy']:.6f}, " +
                    f"Attack Success Rate: {stats['attack_success']:.6f}",
                    end="\r" if self.options.verbosity < 2 else "\n"
                )
            del grads
            del losses
        if self.options.verbosity > 0:
            print()

    def add_clients(self, clients):
        """Add clients to the server"""
        self.num_clients += len(clients)
        self.clients.extend(clients)
