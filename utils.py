"""
Utility functions for use on other classes in this project

Author: Cody Lewis
"""

from typing import NamedTuple
import json
from math import floor

import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
from PIL import Image


class KDD99(Dataset):
    def __init__(self, ds_path, train=True, download=False):
        self.data = torch.tensor([])
        self.targets = torch.tensor([])
        df = pd.read_csv(
            f"{ds_path}/{'train' if train else 'test'}/kddcup.data",
            header=None,
            iterator=True
        )
        nl = 0
        data_len = round(494021 * (0.7 if train else 0.3))
        read_amount = 100_000
        marker = floor(data_len / read_amount) * read_amount
        while read_amount > 0 and (nl := nl + read_amount) <= marker:
            line = df.read(read_amount)
            line = torch.from_numpy(line.to_numpy(np.dtype('float32')))
            self.data = torch.cat((self.data, line[:, 1:-1]))
            self.targets = torch.cat((self.targets, line[:, -1]))
            if nl == marker:
                marker = data_len
                read_amount = data_len % read_amount
        self.len = len(self.data)

    def __getitem__(self, i):
        return (self.data[i], self.targets[i].long())

    def __len__(self):
        return self.len


class Amazon(Dataset):
    def __init__(self, ds_path, train=True, download=False):
        df = pd.read_csv(
            f"{ds_path}/{'train' if train else 'test'}/amazon.data",
            header=None
        )
        data = df.to_numpy(np.dtype('float32'))
        self.data = torch.from_numpy(data[:, :-1])
        self.targets = torch.from_numpy(data[:, -1])
        self.len = len(self.data)

    def __getitem__(self, i):
        return (self.data[i], self.targets[i].long())

    def __len__(self):
        return self.len


class VGGFace(Dataset):
    def __init__(self, ds_path, train=True, download=False, classes=[]):
        self.ds_path = f"{ds_path}/data"
        self.data_paths = []
        self.targets = []
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.train = train
        if train:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        file_info = pd.read_csv(f"{ds_path}/top10_files.csv")
        for _, r in file_info[file_info['train_flag'] == int(not train)].iterrows():
            self.data_paths.append(f"{self.ds_path}/{r['Class_ID']}/{r['file']}")
            self.targets.append(r['Class_ID'])
        self.data_paths = np.array(self.data_paths)
        self.len = len(self.data_paths)
        self.targets = torch.tensor(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = Image.open(self.data_paths[idx])
        X = self.transform(X)
        return (X, self.targets[idx].long())

    def set_data_classes(self, classes):
        ids = torch.arange(self.len)[
            sum([(self.targets == i).long() for i in classes]).bool()]
        self.data_paths = self.data_paths[ids]
        self.targets = self.targets[ids]
        self.len = len(self.data_paths)

    def use_only(self, ids):
        new_ds = VGGFace('./data/vggface', self.train)
        new_ds.data_paths = new_ds.data_paths[ids]
        new_ds.targets = new_ds.targets[ids]
        new_ds.len = len(new_ds.data_paths)
        return new_ds

    def __len__(self):
        return self.len


def load_data(ds_name, train=True, softmax=True):
    """
    Load the specified dataset in a form suitable for the Softmax model

    Keyword arguments:
    ds_name -- name of the dataset
    train -- load the training dataset if true otherwise load the validation
    """
    datasets = {
        "mnist": torchvision.datasets.MNIST,
        "fmnist": torchvision.datasets.FashionMNIST,
        "kddcup99": KDD99,
        "amazon": Amazon,
        "vggface": VGGFace,
    }
    if (chosen_set := datasets.get(ds_name)) is None:
        return torch.tensor(), torch.tensor()
    data = chosen_set(f"./data/{ds_name}", train=train, download=True)
    if softmax:
        X = data.data
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        x_dim = X.shape[-1] if len(X.shape) > 1 else 1
    else:
        x_dim = data[0][0].shape
    Y = data.targets.long().unsqueeze(dim=0)
    return {
        "x": X.float() if softmax else data,
        "y": Y,
        "x_dim": x_dim,
        "y_dim": int(torch.max(Y)) + 1,
    }


def find_stats(model, X, Y, options):
    """Find statistics on the model based on validation data"""
    predictions = torch.argmax(model.predict(X), dim=1)
    accuracy = (predictions == Y[0]).sum().item() / len(Y[0])
    ids = Y[0] == options.adversaries['from']
    attack_success = (
        predictions[ids] == options.adversaries['to']
    ).sum().item() / ids.sum().item()
    return {
        "accuracy": accuracy,
        "attack_success": attack_success
    }


def flatten_grads(grads):
    """Flatten gradients into vectors"""
    flat_grads = []
    for g in grads:
        t = torch.tensor([])
        for p in g['params']:
            t = torch.cat((t, p.flatten()))
        flat_grads.append(t)
    return flat_grads


def flatten_params(params):
    """Flatten params into a vector"""
    flat_params = torch.tensor([])
    for p in params:
        flat_params = torch.cat((flat_params, p.flatten()))
    return flat_params


def write_log(log_fn, stats):
    accuracies = np.mean(np.array(stats['accuracies']), axis=0)
    attack_successes = np.mean(np.array(stats['attack_successes']), axis=0)
    with open(log_fn, "w") as f:
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
    batch_size: int
    learning_rate: list
    lr_changes: list
    architecture: str
    params_mul: int
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

-----[  Server   ]-----
Epochs: {self.server_epochs}
Training algorithm: {self.fit_fun}
Parameters: {self.params}

-----[   Users   ]-----
Amount: {self.users}
Epochs: {self.user_epochs}
Batch size: {self.batch_size}
Learning rates: {self.learning_rate}
Learning rate changing points: {self.lr_changes}

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
            options['batch_size'],
            options['learning_rate'],
            options['lr_changes'],
            options['architecture'],
            options['params_mul'],
            options['fit_fun'],
            options['params'],
            options['adversaries'],
            options['class_shards'],
            options['classes_per_user'],
            options['verbosity'],
            options['result_log_file']
        )
    return None
