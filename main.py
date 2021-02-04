#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entry-point to run the program.

Author: Cody Lewis
"""

import random
import pickle

import torch

from users.adversaries import load_adversary
from users.client import Client
from server import Server
from server.adv_controller import Controller
import utils
import utils.errors
from utils.datasets import load_data
import toggle_stats


def index_match(arr):
    """Check whether the index within an array is equal to the value."""
    for i, a in enumerate(arr):
        if i == a:
            return True
    return False


def find_shards(num_users, num_classes, classes_per_user):
    """Find data class shards according to the parameters."""
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
    """Run a part of the program."""
    try:
        program_flow[current](run_data)
        return run_data
    except utils.errors.MisconfigurationError as e:
        print(f"Miconfiguratation Error: {e}")
        run_data['quit'] = True
        return run_data
    except KeyboardInterrupt:
        print()
        decision = input('Are you sure you want to quit (y/s/n)? ')
        if (dl := decision.lower()).find('y') >= 0:
            run_data['quit'] = True
            return run_data
        elif dl.find('s') >= 0:
            run_data['quit'] = True
            run_data['save'] = True
            return run_data
        return run(program_flow, current, data)


def load_test_data(run_data, train, backdoor):
    return load_data(
        run_data["options"],
        train=train,
        shuffle=False,
        backdoor=backdoor
    )


def system_setup(run_data):
    """Setup the system."""
    run_data["options"] = utils.load_options()
    if run_data["options"].verbosity > 0:
        print("Options set as:")
        print(run_data["options"])
    if 'cuda' in (dev_name := run_data["options"].model_params['device']):
        p = not torch.cuda.is_available()
        c = int(dev_name[dev_name.find(':') + 1:]) + 1
        q = c > torch.cuda.device_count()
        if p or q:
            raise utils.errors.MisconfigurationError(
                f"Device '{dev_name}' is not available on this machine"
            )
    backdoor = run_data['options'].adversaries['type'].find('backdoor') >= 0
    run_data["train_data"] = load_test_data(run_data, True, backdoor)
    run_data["val_data"] = load_test_data(run_data, False, backdoor)
    if backdoor:
        run_data['train_data_no_bd'] = load_test_data(run_data, True, False)
        run_data['val_data_no_bd'] = load_test_data(run_data, False, False)
    run_data['options'].model_params['num_in'] = max(
        run_data["train_data"]['x_dim'],
        run_data["val_data"]['x_dim']
    )
    run_data['options'].model_params['num_out'] = max(
        run_data["train_data"]['y_dim'],
        run_data["val_data"]['y_dim']
    )
    run_data['sim_number'] = 0
    if run_data['options'].adversaries['optimized']:
        run_data['sybil_controller'] = Controller(run_data['options'])
        run_data['controller_toggle'] = []
    return run_data


def setup_users(run_data):
    """Setup the users/clients for the system."""
    run_data["user_classes"] = [
        Client if i <= run_data["options"].users * (
            1 - run_data["options"].adversaries['percent_adv'])
        else load_adversary(
            run_data["options"],
            run_data.get('sybil_controller')
        )
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
    """Run the simulations."""
    for i in range(run_data['sim_number'], run_data["options"].num_sims):
        print(f"Simulation {i + 1}/{run_data['options'].num_sims}")
        if not run_data.get('sim_setup'):
            run_data["server"] = Server(
                run_data['options'].model_params['num_in'],
                run_data['options'].model_params['num_out'],
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
            if (vd := run_data.get('val_data_no_bd')) is not None:
                run_data["server"].fit(
                    vd['dataloader'],
                    run_data['epoch'],
                    run_data["options"].server_epochs,
                    run_data["val_data"]['dataloader'],
                    run_data.get('sybil_controller')
                )
            else:
                run_data["server"].fit(
                    run_data["val_data"]['dataloader'],
                    run_data['epoch'],
                    run_data["options"].server_epochs,
                    syb_con=run_data.get('sybil_controller')
                )
        confusion_matrices, cm_bd = run_data['server'].get_conf_matrices()
        if (scm := run_data.get("sim_confusion_matrices")) is not None:
            scm += confusion_matrices
        else:
            run_data["sim_confusion_matrices"] = confusion_matrices
        if cm_bd is not None:
            if (scmb := run_data.get("sim_confusion_matrices_bd")) is not None:
                scmb += cm_bd
            else:
                run_data["sim_confusion_matrices_bd"] = cm_bd
        if (con := run_data.get('sybil_controller')) is not None:
            run_data['controller_toggle'].append(con.toggle_record.copy())
            con.setup()
        run_data['sim_setup'] = False
        run_data['sim_number'] += 1
        if run_data["options"].verbosity > 0:
            print()
            print("Done training.")
        criterion = torch.nn.CrossEntropyLoss()
        loss, stats = gen_conf(
            run_data,
            criterion,
            "train_data"
        )
        loss_val, stats_val = gen_conf(
            run_data,
            criterion,
            "val_data"
        )
        if run_data.get('train_data_no_bd') is not None:
            loss_nb, stats_nb = gen_conf(
                run_data,
                criterion,
                "train_data_no_bd"
            )
            loss_val_nb, stats_val_nb = gen_conf(
                run_data,
                criterion,
                "val_data_no_bd"
            )
            print()
            print("Normal Stats:")
            print(get_printable_stats(loss_nb, loss_val_nb, stats_nb, stats_val_nb))
            print("Backdoored Stats:")
        print(get_printable_stats(loss, loss_val, stats, stats_val))
    return run_data


def get_printable_stats(loss, loss_val, stats, stats_val):
    return f"Loss: t: {loss}, v: {loss_val}\n" + \
        f"Accuracy: t: {stats['accuracy']:%}, " + \
        f"v: {stats_val['accuracy']:%}\n" + \
        f"MCC: t: {stats['MCC']}, v: {stats_val['MCC']}\n" + \
        f"Attack success rate: t: {stats['attack_success']:%}, " + \
        f"v: {stats_val['attack_success']:%}\n"


def gen_conf(run_data, criterion, dl_name):
    loss, conf_mat = utils.gen_confusion_matrix(
        run_data["server"].net,
        run_data[dl_name]['dataloader'],
        criterion,
        run_data["server"].nb_classes,
        run_data["options"]
    )
    stats = utils.gen_conf_stats(conf_mat, run_data["options"])
    return loss, stats

def write_results(run_data):
    """Write all of the recorded results from the experiments"""
    if run_data["options"].verbosity > 0:
        print()
        print(f"Writing confusion matrices to {run_data['options'].result_file}...")
    utils.write_results(
        run_data["options"].result_file,
        torch.round(
            run_data["sim_confusion_matrices"] / run_data["options"].num_sims
        ).long()
    )
    if (scmb := run_data.get('sim_confusion_matrices_bd')) is not None:
        utils.write_results(
            f"bd_{run_data['options'].result_file}",
            torch.round(
                scmb / run_data["options"].num_sims
            ).long()
        )
    if (ct := run_data.get('controller_toggle')) is not None:
        tr_fn = 'toggle_record.csv'
        print(f"Writing toggle record to {tr_fn}...")
        max_len = max([len(a) for a in ct])
        for a in ct:
            while len(a) < max_len:
                a.append(-1)
        toggle_stats.write_results(
            run_data["options"], torch.tensor(ct), tr_fn
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
            if data.get('save'):
                run(program_flow, 'write_results', data)
            print("bye.")
            break
    del data
    # sys.exit(0)
