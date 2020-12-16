"""
Classes and functions for the server networking aspect of federated learning

Author: Cody Lewis
"""


import time

import torch
import torch.nn as nn

from server.global_model import GlobalModel
import utils


def gen_conf_mat_and_stats(server, dataloader):
    loss, confusion_matrix = utils.gen_confusion_matrix(
        server.net,
        dataloader,
        server.criterion,
        server.nb_classes,
        server.options
    )
    stats = utils.gen_conf_stats(confusion_matrix, server.options)
    return loss, confusion_matrix, stats


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
        if options.adversaries['type'].find('backdoor') >= 0:
            self.confusion_matrices_bd = torch.tensor([], dtype=int)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, dataloader, e, epochs, dataloader_bd=None, syb_con=None):
        start = time.time()
        grads = []
        net_params = self.net.get_params()
        for c in self.clients:
            c.net.copy_params(net_params)
            grads.append(c.fit()[1])
        # Have a sybil controller intercept and copy the gradients
        if syb_con is not None:
            attack = syb_con.intercept(grads)
        self.net.fit(grads, self.options.params)
        loss, confusion_matrix, stats = gen_conf_mat_and_stats(self, dataloader)
        self.confusion_matrices = torch.cat(
            (self.confusion_matrices, confusion_matrix.unsqueeze(dim=0))
        )
        if dataloader_bd is not None:
            loss_bd, confusion_matrix, stats_bd = gen_conf_mat_and_stats(
                self,
                dataloader_bd
            )
            self.confusion_matrices_bd = torch.cat(
                (self.confusion_matrices_bd, confusion_matrix.unsqueeze(dim=0))
            )
        if self.options.verbosity > 0:
            print(
                f"[  E: {e + 1:=}/{epochs}, " +
                f"L: {loss: =7.5f}, " +
                f"Acc: {stats['accuracy']: =7.5f}, " +
                f"ASR: {stats['attack_success']: =7.5f}, " +
                (f"BD: {stats_bd['attack_success']: =7.5f}, " if dataloader_bd is
                    not None else '') +
                f"T: {time.time() - start: =7.5f}s " +
                ('I' if syb_con is not None and attack else ' ') +
                "]",
                end="\r" if self.options.verbosity < 2 else "\n"
            )
        del grads

    def add_clients(self, clients):
        """Add clients to the server"""
        self.num_clients += len(clients)
        self.clients.extend(clients)

    def get_conf_matrices(self):
        if self.options.adversaries['type'].find('backdoor') >= 0:
            return self.confusion_matrices, self.confusion_matrices_bd
        return self.confusion_matrices, None
