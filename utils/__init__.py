"""
Utility functions for use on other classes in this project

Author: Cody Lewis
"""

from typing import NamedTuple
import math
import json

import torch
from sklearn.metrics import confusion_matrix as cm

import numpy as np


def gen_confusion_matrix(model, dataloader, criterion, nb_classes, options):
    """
    Give the loss of the model across the data and Generate a confusion matrix
    """
    with torch.no_grad():
        loss = 0
        denom = 0
        confusion_matrix = torch.zeros(nb_classes, nb_classes, dtype=int)
        for x, y in dataloader:
            x = x.to(options.model_params['device'])
            y = y.to(options.model_params['device'])
            predictions = model.predict(x)
            loss += criterion(predictions, y)
            denom += len(y)
            predictions = torch.argmax(predictions, dim=1)
            confusion_matrix += torch.from_numpy(
                cm(predictions.cpu(), y.cpu(), labels=np.arange(nb_classes))
            )
        return loss / denom * 100, confusion_matrix


def gen_conf_stats(confusion_matrix, options):
    """Find some statistics based on the given confusion_matrix"""
    accuracy = 0
    total = 0
    attack_success_n = 0
    attack_success_d = 0
    class_acc = [[0, 0] for _ in confusion_matrix]
    t = torch.tensor([0 for _ in confusion_matrix])
    p = torch.tensor([0 for _ in confusion_matrix])
    for x, row in enumerate(confusion_matrix):
        for y, cell in enumerate(row):
            # cell = int(cell)
            cell.item()
            t[y] += cell
            p[x] += cell
            if x == y:
                accuracy += cell
                class_acc[y][0] += cell
            if y == options.adversaries['from']:
                if x == options.adversaries['to']:
                    attack_success_n += cell
                attack_success_d += cell
            total += cell
            class_acc[y][1] += cell
    f = lambda x, y: x / y if y > 0 else 0.
    stats = {
        "accuracy": f(accuracy, total),
        "attack_success": f(attack_success_n, attack_success_d),
        "MCC": f((accuracy * total - t.dot(p)),
            (math.sqrt(total**2 - p.dot(p)) * math.sqrt(total**2 - t.dot(t))))
    }
    for i, acc in enumerate(class_acc):
        stats[f"accuracy_{i}"] = f(acc[0], acc[1])
    return stats


def gen_experiment_stats(sim_confusion_matrices, options):
    """Find the statistics across multiple simulations"""
    stats = merge_dicts(
        [gen_sim_stats(c, options) for c in sim_confusion_matrices]
    )
    for k, v in stats.items():
        stats[k] = torch.tensor(v)
    return stats


def gen_sim_stats(confusion_matrices, options):
    """Find the stastics of one simulation"""
    return merge_dicts(
        [gen_conf_stats(c, options) for c in confusion_matrices]
    )


def merge_dicts(dict_list):
    """Merge two dictionaries"""
    merged = {k: [] for k in dict_list[0].keys()}
    for d in dict_list:
        for k, v in d.items():
            merged[k].append(v)
    return merged


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


def write_results(result_file, confusion_matrices):
    """Write the results of an experiment to a pickle file"""
    torch.save(confusion_matrices, result_file)


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
    result_file: str

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
Results file: {self.result_file}

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
            options['result_file']
        )
    return None
