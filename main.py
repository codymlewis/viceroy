#!/usr/bin/env python3

"""
Entry-point to run the program

Author: Cody Lewis
"""

import random

import torch

from adversaries import ADVERSARY_TYPES
from client import Client
import errors
from server import Server
import utils
from datasets import load_data


def index_match(arr):
    """Check whether the index within an array is equal to the value"""
    for i, a in enumerate(arr):
        if i == a:
            return True
    return False


def find_shards(num_users, num_classes, classes_per_user):
    """Find data class shards according to the parameters"""
    end_halves = [[i for i in range(num_classes)]
            for _ in range(classes_per_user - 1)]
    if num_classes / classes_per_user < num_users:
        for end_half in end_halves:
            while index_match(end_half):
                random.shuffle(end_half)
    return [
        [
            i % num_classes,
            (num_classes - i - 1) % num_classes
            if i < num_users / classes_per_user else end_halves[0][i % num_classes]
            ] + ([eh[i % num_classes] for eh in end_halves[1:]] if
            classes_per_user > 2 else [])
        for i in range(num_users)
    ]


if __name__ == '__main__':
    try:
        options = utils.load_options()
        if options.verbosity > 0:
            print("Options set as:")
            print(options)
        if 'cuda' in (dev_name := options.model_params['device']):
            p = not torch.cuda.is_available()
            c = int(dev_name[dev_name.find(':') + 1:]) + 1
            q = c > torch.cuda.device_count()
            if p or q:
                raise errors.MisconfigurationError(
                    f"Device '{dev_name}' is not available on this machine"
                )
        train_data = load_data(
            options,
            train=True,
            shuffle=False,
        )
        val_data = load_data(
            options,
            train=False,
            shuffle=False,
        )
        user_classes = [
            Client if i <= options.users * (1 - options.adversaries['percent_adv'])
            else ADVERSARY_TYPES[options.adversaries['type']]
            for i in range(1, options.users + 1)
        ]
        if options.class_shards:
            class_shards = options.class_shards
        else:
            class_shards = find_shards(
                options.users,
                val_data['y_dim'],
                options.classes_per_user
            )
        if options.class_shards is None and options.verbosity > 0:
            print("Assigned class shards:")
            print(class_shards)
            print()
        experiment_stats = {"accuracies": [], "attack_successes": []}
        for i in range(options.num_sims):
            print(f"Simulation {i + 1}/{options.num_sims}")
            server = Server(val_data['x_dim'], val_data['y_dim'], options)
            server.add_clients(
                [
                    u(
                        options,
                        class_shards[i]
                    ) for i, u in enumerate(user_classes)
                ]
            )
            print("Starting training...")
            accuracies, attack_successes = server.fit(
                val_data['dataloader'], options.server_epochs
            )
            experiment_stats['accuracies'].append(accuracies)
            experiment_stats['attack_successes'].append(attack_successes)
            if options.verbosity > 0:
                print("Done training.")
            criterion = torch.nn.CrossEntropyLoss()
            stats = utils.find_stats(
                server.net, train_data['dataloader'], criterion, options
            )
            stats_val = utils.find_stats(
                server.net, val_data['dataloader'], criterion, options
            )
            print(f"Loss: t: {stats['loss']}, v: {stats_val['loss']}")
            print(f"Accuracy: t: {stats['accuracy'] * 100}%, ", end="")
            print(f"v: {stats_val['accuracy'] * 100}%")
            print(
                f"Attack success rate: t: {stats['attack_success'] * 100}%, ",
                end=""
            )
            print(f"v: {stats_val['attack_success'] * 100}%")
            print()
        if options.verbosity > 0:
            print(f"Writing averaged results to {options.result_log_file}...")
        utils.write_log(options.result_log_file, experiment_stats)
        if options.verbosity > 0:
            print("Done.")
    except errors.MisconfigurationError as e:
        print(f"Miconfiguratation Error: {e}")
