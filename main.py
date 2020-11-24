#!/usr/bin/env python3

"""
Entry-point to run the program

Author: Cody Lewis
"""

import random

import torch
import numpy as np

from adversaries import load_adversary
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
    end_halves = [list(range(num_classes))
                  for _ in range(classes_per_user - 1)]
    if num_classes / classes_per_user < num_users:
        for end_half in end_halves:
            while index_match(end_half):
                random.shuffle(end_half)
    return [
        [
            i % num_classes,
            (num_classes - i - 1) % num_classes
            if i < num_users / classes_per_user else
            end_halves[0][i % num_classes]
        ] + ([eh[i % num_classes] for eh in end_halves[1:]] if
             classes_per_user > 2 else [])
        for i in range(num_users)
    ]


def run(program_flow, current, run_data):
    try:
        program_flow[current](run_data)
        return run_data
    except errors.MisconfigurationError as e:
        print(f"Miconfiguratation Error: {e}")
    except KeyboardInterrupt:
        print()
        decision = input('Are you sure you want to quit? ')
        if decision.lower().find('y') >= 0:
            run_data['quit'] = True
            return run_data
        return run(program_flow, current, data)


def system_setup(run_data):
    run_data["options"] = utils.load_options()
    if run_data["options"].verbosity > 0:
        print("Options set as:")
        print(run_data["options"])
    if 'cuda' in (dev_name := run_data["options"].model_params['device']):
        p = not torch.cuda.is_available()
        c = int(dev_name[dev_name.find(':') + 1:]) + 1
        q = c > torch.cuda.device_count()
        if p or q:
            raise errors.MisconfigurationError(
                f"Device '{dev_name}' is not available on this machine"
            )
    run_data["train_data"] = load_data(
        run_data["options"],
        train=True,
        shuffle=False,
    )
    run_data["val_data"] = load_data(
        run_data["options"],
        train=False,
        shuffle=False,
    )
    run_data['sim_number'] = 0
    return run_data

def setup_users(run_data):
    run_data["user_classes"] = [
        Client if i <= run_data["options"].users * (
            1 - run_data["options"].adversaries['percent_adv'])
        else load_adversary(run_data["options"].adversaries['type'])
        for i in range(1, run_data["options"].users + 1)
    ]
    if run_data["options"].class_shards:
        run_data["class_shards"] = run_data["options"].class_shards
    else:
        run_data["class_shards"] = find_shards(
            run_data["options"].users,
            run_data["val_data"]['y_dim'],
            run_data["options"].classes_per_user
        )
    if run_data["options"].class_shards is None and \
            run_data["options"].verbosity > 0:
        print("Assigned class shards:")
        print(run_data["class_shards"])
        print()
    return run_data


def run_simulations(run_data):
    run_data["sim_confusion_matrices"] = torch.tensor([], dtype=int)
    for i in range(run_data['sim_number'], run_data["options"].num_sims):
        print(f"Simulation {i + 1}/{run_data['options'].num_sims}")
        if not run_data.get('sim_setup'):
            run_data["server"] = Server(
                max(
                    run_data["train_data"]['x_dim'],
                    run_data["val_data"]['x_dim']
                ),
                max(
                    run_data["train_data"]['y_dim'],
                    run_data["val_data"]['y_dim']
                ),
                run_data["options"]
            )
            run_data["server"].add_clients(
                [
                    u(
                        run_data["options"],
                        run_data["class_shards"][i]
                    ) for i, u in enumerate(run_data["user_classes"])
                ]
            )
            run_data['sim_setup'] = True
            run_data['epoch'] = 0
        print("Starting training...")
        for run_data['epoch'] in range(run_data['epoch'],
                run_data['options'].server_epochs):
            run_data["server"].fit(
                run_data["val_data"]['dataloader'],
                run_data['epoch'],
                run_data["options"].server_epochs
            )
        confusion_matrices = run_data['server'].get_conf_matrices()
        if run_data["options"].verbosity > 0:
            print()
        run_data["sim_confusion_matrices"] = torch.cat(
            (
                run_data["sim_confusion_matrices"],
                confusion_matrices.unsqueeze(dim=0)
            )
        )
        print()
        run_data['sim_setup'] = False
        run_data['sim_number'] += 1
        if run_data["options"].verbosity > 0:
            print("Done training.")
        criterion = torch.nn.CrossEntropyLoss()
        loss, conf_mat = utils.gen_confusion_matrix(
            run_data["server"].net,
            run_data["train_data"]['dataloader'],
            criterion,
            run_data["server"].nb_classes,
            run_data["options"]
        )
        stats = utils.gen_conf_stats(conf_mat, run_data["options"])
        loss_val, conf_mat = utils.gen_confusion_matrix(
            run_data["server"].net,
            run_data["val_data"]['dataloader'],
            criterion,
            run_data["server"].nb_classes,
            run_data["options"]
        )
        stats_val = utils.gen_conf_stats(conf_mat, run_data["options"])
        print(f"Loss: t: {loss}, v: {loss_val}")
        print(f"Accuracy: t: {stats['accuracy'] * 100}%, ", end="")
        print(f"v: {stats_val['accuracy'] * 100}%")
        print(f"MCC: t: {stats['MCC']}, v: {stats_val['MCC']}")
        print(
            f"Attack success rate: t: {stats['attack_success'] * 100}%, ",
            end=""
        )
        print(f"v: {stats_val['attack_success'] * 100}%")
    return run_data


def write_results(run_data):
    if run_data["options"].verbosity > 0:
        print()
        print(f"Writing confusion matrices to {run_data['options'].result_file}...")
    utils.write_results(
        run_data["options"].result_file,
        run_data["sim_confusion_matrices"]
    )
    if run_data["options"].verbosity > 0:
        print("Done.")
    return run_data



if __name__ == '__main__':
    program_flow = {
        "system_setup": system_setup,
        "setup_users": setup_users,
        "run_simulations": run_simulations,
        "write_results": write_results
    }
    data = {"quit": False}
    for k in program_flow.keys():
        data = run(program_flow, k, data)
        if data['quit']:
            print("bye.")
            break
