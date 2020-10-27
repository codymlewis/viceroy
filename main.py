#!/usr/bin/env python3

"""
Entry-point to run the program

Author: Cody Lewis
"""


import random

from adversaries import ADVERSARY_TYPES
from client import Client
from server import Server
import utils


def main():
    """Function containing the main program flow"""
    options = utils.load_options()
    if options.verbosity > 0:
        print("Loading Datasets and Creating Models...")
    train_data = utils.load_data("mnist", train=True)
    val_data = utils.load_data("mnist", train=False)
    server = Server(val_data['x_dim'], val_data['y_dim'], options)
    stats = utils.find_stats(server.net, val_data['x'], val_data['y'], options)
    if options.verbosity > 1:
        print(f"Creating log file: {options.result_log_file}...")
    utils.create_log(options.result_log_file, stats)
    if options.verbosity > 1:
        print("Done log file creation.")
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
        print("Done dataset and model creation.")
    print("Starting training...")
    server.fit(val_data['x'], val_data['y'], options.server_epochs)
    if options.verbosity > 0:
        print("Done training.")
    print()
    print("-----[Results]-----")
    stats = utils.find_stats(server.net, val_data['x'], val_data['y'], options)
    print(f"Accuracy: {stats['accuracy'] * 100}%")
    print(f"Attack Success Rate: {stats['attack_success'] * 100}%")
    print("-------------------")


if __name__ == '__main__':
    main()
