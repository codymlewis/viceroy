"""
Classes and functions for the server networking aspect of federated learning

Author: Cody Lewis
"""


import time

import torch
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
        self.nb_classes = num_out
        self.options = options

    def fit(self, dataloader, epochs):
        confusion_matrices = torch.tensor([], dtype=int)
        criterion = nn.CrossEntropyLoss()
        for e in range(epochs):
            start = time.time()
            grads = []
            for c in self.clients:
                c.net.copy_params(self.net.get_params())
                grads.append(c.fit()[1])
            self.net.fit(grads, self.options.params)
            loss, confusion_matrix = utils.gen_confusion_matrix(
                self.net,
                dataloader,
                criterion,
                self.nb_classes,
                self.options
            )
            confusion_matrices = torch.cat(
                (confusion_matrices, confusion_matrix.unsqueeze(dim=0))
            )
            stats = utils.gen_conf_stats(confusion_matrix, self.options)
            if self.options.verbosity > 0:
                print(
                    f"[ E: {e + 1}/{epochs}, " +
                    f"L: {loss:.6f}, " +
                    f"Acc: {stats['accuracy']:.6f}, " +
                    f"ASR: {stats['attack_success']:.6f}, " +
                    f"T: {time.time() - start:.6f}s ]",
                    end="\r" if self.options.verbosity < 2 else "\n"
                )
            del grads
        if self.options.verbosity > 0:
            print()
        return confusion_matrices

    def add_clients(self, clients):
        """Add clients to the server"""
        self.num_clients += len(clients)
        self.clients.extend(clients)
