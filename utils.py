"""
Utility functions for use on other classes in this project

Author: Cody Lewis
"""

from typing import NamedTuple
import json

import torch

import numpy as np


def find_stats(model, dataloader, criterion, options):
    """Find statistics on the model based on validation data"""
    denom = 0
    as_denom = 0
    accuracy = 0
    attack_success = 0
    loss = 0
    for x, y in dataloader:
        with torch.no_grad():
            x = x.to(options.model_params['device'])
            y = y.to(options.model_params['device'])
            predictions = model.predict(x)
            loss += criterion(predictions, y)
            predictions = torch.argmax(predictions, dim=1)
            accuracy += (predictions == y).sum().item()
            ids = y == options.adversaries['from']
            attack_success += (
                predictions[ids] == options.adversaries['to']
            ).sum().item()
            denom += len(y)
            as_denom += ids.sum().item()
    return {
        "accuracy": accuracy / denom,
        "attack_success": attack_success / as_denom,
        "loss": loss / denom * 100,
    }


def flatten_grads(grads, params):
    """Flatten gradients into vectors"""
    with torch.no_grad():
        flat_grads = []
        for g in grads:
            t = torch.tensor([]).to(params['device'])
            for p in g['params']:
                t = torch.cat((t, p.flatten()))
            flat_grads.append(t)
        return flat_grads


def flatten_params(params, options):
    """Flatten params into a vector"""
    with torch.no_grad():
        flat_params = torch.tensor([]).to(options['device'])
        for p in params:
            flat_params = torch.cat((flat_params, p.flatten()))
        return flat_params


def write_log(log_file_name, stats):
    accuracies = np.mean(np.array(stats['accuracies']), axis=0)
    attack_successes = np.mean(np.array(stats['attack_successes']), axis=0)
    with open(log_file_name, "w") as f:
        f.write("epoch,accuracy,attack_success\n")
        for i, (a, b) in enumerate(zip(accuracies, attack_successes)):
            f.write(f"{i},{a},{b}\n")


class Options(NamedTuple):
    """Structure out the data from the options file"""
    dataset: str
    num_sims: int
    server_epochs: int
    user_epochs: int
    users: int
    model_params: dict
    fit_fun: str
    params: dict
    adversaries: dict
    class_shards: list
    classes_per_user: int
    verbosity: int
    result_log_file: str

    def __str__(self):
        new_line = '\n'
        if self.class_shards:
            shard_str = new_line.join([f"{v}" for v in self.class_shards])
        else:
            shard_str = f"Randomly assigned with {self.classes_per_user} classes per user"
        return f"""
-----[  General  ]-----
Dataset: {self.dataset}
Number of simulations: {self.num_sims}
Verbosity: {self.verbosity}
Log file: {self.result_log_file}

-----[   Model   ]-----
{new_line.join([f"{k}: {v}" for k, v in self.model_params.items()])}

-----[  Server   ]-----
Epochs: {self.server_epochs}
Training algorithm: {self.fit_fun}
Parameters: {self.params}

-----[   Users   ]-----
Amount: {self.users}
Epochs: {self.user_epochs}

-----[Adversaries]-----
{new_line.join([f"{k}: {v}" for k, v in self.adversaries.items()])}

-----[Data Shards]-----
{shard_str}

-----------------------
"""


def load_options():
    """Load a structure containing the options"""
    with open("options.json", "r") as f:
        options = json.loads(f.read())
        return Options(
            options['dataset'],
            options['num_sims'],
            options['server_epochs'],
            options['user_epochs'],
            options['users'],
            options['model_params'],
            options['fit_fun'],
            options['params'],
            options['adversaries'],
            options['class_shards'],
            options['classes_per_user'],
            options['verbosity'],
            options['result_log_file']
        )
    return None
