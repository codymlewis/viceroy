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


def index_match(arr):
    """Check whether the index within an array is equal to the value"""
    for i, a in enumerate(arr):
        if i == a:
            return True
    return False


def find_shards(num_users, percent_adv, num_classes):
    """Find data class shards according to the parameters"""
    end_half = [i for i in range(num_classes)]
    if num_classes / 2 < num_users:
        while index_match(end_half):
            random.shuffle(end_half)
    return [
        [
            i % num_classes,
            (num_classes - i - 1) % num_classes
            if i < num_users / 2 else end_half[i % num_classes]
        ]
        for i in range(num_users)
    ]


if __name__ == '__main__':
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
    user_classes = [
        Client if i <= options.users * (1 - options.adversaries['percent_adv'])
        else ADVERSARY_TYPES[options.adversaries['type']]
        for i in range(options.users)
    ]
    if options.class_shards:
        class_shards = options.class_shards
    else:
        class_shards = find_shards(
            options.users,
            options.adversaries['percent_adv'],
            val_data['y_dim']
        )
    if options.verbosity > 0:
        print("Assigned class shards:")
        print(class_shards)
    server.add_clients(
        [
            u(
                train_data,
                options,
                class_shards[i]
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
    stats = utils.find_stats(server.net, train_data['x'],
                             train_data['y'], options)
    print("Training:")
    print(f"Accuracy: {stats['accuracy'] * 100}%")
    print(f"Attack Success Rate: {stats['attack_success'] * 100}%")
    stats = utils.find_stats(server.net, val_data['x'], val_data['y'], options)
    print("Validation")
    print(f"Accuracy: {stats['accuracy'] * 100}%")
    print(f"Attack Success Rate: {stats['attack_success'] * 100}%")
    print("-------------------")
