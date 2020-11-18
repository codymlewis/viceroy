"""
Classes and functions for the server networking aspect of federated learning

Author: Cody Lewis
"""

import torch.nn as nn

from global_model import GlobalModel
import utils


class Server:
    """Federated learning server class"""
    def __init__(self, num_in, num_out, options):
        self.net = GlobalModel(
            num_in,
            num_out,
            options,
        )
        self.num_clients = 0
        self.clients = []
        self.options = options

    def fit(self, dataloader, epochs):
        accuracies, attack_successes = [], []
        criterion = nn.CrossEntropyLoss()
        for e in range(epochs):
            grads = []
            for c in self.clients:
                c.net.copy_params(self.net.get_params())
                grads.append(c.fit()[1])
            self.net.fit(grads, self.options.params)
            stats = utils.find_stats(
                self.net, dataloader, criterion, self.options
            )
            accuracies.append(stats['accuracy'])
            attack_successes.append(stats['attack_success'])
            if self.options.verbosity > 0:
                print(
                    f"Epoch: {e + 1}/{epochs}, " +
                    f"Loss: {stats['loss']:.6f}, " +
                    f"Accuracy: {stats['accuracy']:.6f}, " +
                    f"Attack Success Rate: {stats['attack_success']:.6f}",
                    end="\r" if self.options.verbosity < 2 else "\n"
                )
            del grads
        if self.options.verbosity > 0:
            print()
        return accuracies, attack_successes

    def add_clients(self, clients):
        """Add clients to the server"""
        self.num_clients += len(clients)
        self.clients.extend(clients)
