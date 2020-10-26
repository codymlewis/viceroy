"""
Classes and functions for the server networking aspect of federated learning

Author: Cody Lewis
"""

import random
import threading

import torch.nn as nn

from client import Client
from adversaries import ADVERSARY_TYPES
from global_model import GlobalModel
import utils

# TODO: Maybe verbosity for log file writing


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
            grads = dict()
            losses = dict()
            threads = list()
            for c in self.clients:
                c.net.copy_params(self.net.get_params())
                t = threading.Thread(target=client_fit, args=(c,))
                threads.append(t)
                t.start()
            for i, (t, c) in enumerate(zip(threads, self.clients)):
                t.join()
                grads[i] = c.latest_grad
                losses[i] = c.latest_loss
            self.net.fit(grads, self.options.params)
            stats = utils.find_stats(self.net, X, Y, self.options)
            utils.log_stats(self.options.result_log_file, stats)
            if self.options.verbosity > 0:
                print(
                    f"Epoch: {e + 1}/{epochs}, " +
                    f"Loss: {criterion(self.net.predict(X), Y[0]):.6f}, " +
                    f"Accuracy: {stats['accuracy']:.6f}, " +
                    f"Attack Success Rate: {stats['attack_success']:.6f}",
                    end="\r" if self.options.verbosity < 2 else "\n"
                )
        if self.options.verbosity > 0:
            print()

    def add_clients(self, clients):
        """Add clients to the server"""
        self.num_clients += len(clients)
        self.clients.extend(clients)


def client_fit(client):
    """Function that fits a client, for use within a thread"""
    client.fit_async(verbose=False)


if __name__ == '__main__':
    options = utils.load_options()
    if options.verbosity > 0:
        print("Loading Datasets and Creating Models...")
    train_data = utils.load_data("mnist", train=True)
    val_data = utils.load_data("mnist", train=False)
    server = Server(val_data['x_dim'], val_data['y_dim'], options)
    stats = utils.find_stats(server.net, val_data['x'], val_data['y'], options)
    utils.create_log(options.result_log_file, stats)
    if options.users >= val_data['y_dim']:
        class_shards = [
            i % val_data['y_dim'] for i in range(
                2 * (options.users - (options.users % val_data['y_dim']))
            )
        ]
        if options.users % val_data['y_dim']:
            class_shards += random.sample(
                [i for i in range(val_data['y_dim'])],
                options.users % val_data['y_dim']
            )
    user_classes = [
        Client if i < options.users * (1 - options.adversaries['percent_adv'])
        else ADVERSARY_TYPES[options.adversaries['type']]
        for i in range(options.users)
    ]
    server.add_clients(
        [
            u(
                train_data,
                options,
                class_shards[2*i:2*i + 2]
            ) for i, u in enumerate(user_classes)
        ]
    )
    if options.verbosity > 0:
        print("Done.")
    print("Starting training...")
    server.fit(val_data['x'], val_data['y'], options.server_epochs)
    if options.verbosity > 0:
        print("Done.")
    print()
    print("-----[Results]-----")
    stats = utils.find_stats(server.net, val_data['x'], val_data['y'], options)
    print(f"Accuracy: {stats['accuracy'] * 100}%")
    print(f"Attack Success Rate: {stats['attack_success'] * 100}%")
    print("-------------------")
