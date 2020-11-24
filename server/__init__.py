"""
Classes and functions for the server networking aspect of federated learning

Author: Cody Lewis
"""


import time

import torch
import torch.nn as nn

from server.global_model import GlobalModel
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
        self.confusion_matrices = torch.tensor([], dtype=int)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, dataloader, e, epochs):
        start = time.time()
        grads = []
        for c in self.clients:
            c.net.copy_params(self.net.get_params())
            grads.append(c.fit()[1])
        self.net.fit(grads, self.options.params)
        loss, confusion_matrix = utils.gen_confusion_matrix(
            self.net,
            dataloader,
            self.criterion,
            self.nb_classes,
            self.options
        )
        self.confusion_matrices = torch.cat(
            (self.confusion_matrices, confusion_matrix.unsqueeze(dim=0))
        )
        stats = utils.gen_conf_stats(confusion_matrix, self.options)
        if self.options.verbosity > 0:
            print(
                f"[ E: {e + 1}/{epochs}, " +
                f"L: {loss:.6f}, " +
                f"Acc: {stats['accuracy']:.6f}, " +
                f"MCC: {stats['MCC']:.6f}, " +
                f"ASR: {stats['attack_success']:.6f}, " +
                f"T: {time.time() - start:.6f}s ]",
                end="\r" if self.options.verbosity < 2 else "\n"
            )
        del grads

    def add_clients(self, clients):
        """Add clients to the server"""
        self.num_clients += len(clients)
        self.clients.extend(clients)

    def get_conf_matrices(self):
        return self.confusion_matrices
