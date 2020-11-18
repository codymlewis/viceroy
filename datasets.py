from math import floor
from abc import abstractmethod

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import pandas as pd
from PIL import Image

import errors


class DatasetWrapper(Dataset):
    def __init__(self):
        self.targets = torch.tensor([])
        self.y_dim = 0

    def __len__(self):
        return len(self.targets)

    @abstractmethod
    def __getitem__(self, i):
        pass

    def get_dims(self):
        if len(self) < 1:
            return (0, 0)
        x, _ = self[0]
        return (x.shape[0], self.y_dim)

    def get_idx(self, classes):
        return torch.arange(len(self.targets))[
            sum([(self.targets == i).long() for i in classes]).bool()
        ]

    def assign_to_classes(self, classes):
        idx = self.get_idx(classes)
        self.data = self.data[idx]
        self.targets = self.targets[idx]


class MNIST(DatasetWrapper):
    def __init__(self, ds_path, train=True, download=False, classes=[]):
        super().__init__()
        ds = torchvision.datasets.MNIST(
            ds_path,
            train=train,
            download=download
        )
        self.data = ds.data.flatten(1).float()
        self.targets = ds.targets
        self.y_dim = len(self.targets.unique())
        if classes:
            self.assign_to_classes(classes)

    def __getitem__(self, i):
        return (self.data[i], self.targets[i])


class FashionMNIST(DatasetWrapper):
    def __init__(self, ds_path, train=True, download=False, classes=[]):
        super().__init__()
        ds = torchvision.datasets.MNIST(
            ds_path,
            train=train,
            download=download
        )
        self.data = ds.data.flatten(1).float()
        self.targets = ds.targets
        self.y_dim = len(self.targets.unique())
        if classes:
            self.assign_to_classes(classes)

    def __getitem__(self, i):
        return (self.data[i], self.targets[i])


class KDD99(DatasetWrapper):
    def __init__(self, ds_path, train=True, download=False, classes=[]):
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
        self.y_dim = len(self.targets.unique())
        if classes:
            self.assign_to_classes(classes)

    def __getitem__(self, i):
        return (self.data[i], self.targets[i].long())


class Amazon(DatasetWrapper):
    def __init__(self, ds_path, train=True, download=False, classes=[]):
        df = pd.read_csv(
            f"{ds_path}/{'train' if train else 'test'}/amazon.data",
            header=None
        )
        data = df.to_numpy(np.dtype('float32'))
        self.data = torch.from_numpy(data[:, :-1])
        self.targets = torch.from_numpy(data[:, -1])
        self.y_dim = len(self.targets.unique())
        if classes:
            self.assign_to_classes(classes)

    def __getitem__(self, i):
        return (self.data[i], self.targets[i].long())


class VGGFace(DatasetWrapper):
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
        unique_classes = set()
        for _, r in file_info[file_info['train_flag'] == int(not train)].iterrows():
            if r['Class_ID'] not in unique_classes:
                unique_classes = unique_classes.union({r['Class_ID']})
            if not classes or r['Class_ID'] in classes:
                self.data_paths.append(f"{self.ds_path}/{r['Class_ID']}/{r['file']}")
                self.targets.append(r['Class_ID'])
        self.y_dim = len(unique_classes)
        self.data_paths = np.array(self.data_paths)
        self.targets = torch.tensor(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = Image.open(self.data_paths[idx])
        X = self.transform(X)
        return (X, self.targets[idx].long())


def load_data(options, train=True, shuffle=True, classes=[]):
    """
    Load the specified dataset in a form suitable for the model

    Keyword arguments:
    options -- options for the simulation
    train -- load the training dataset if true otherwise load the validation
    classes -- use only the classes in list, use all classes if empty list
    """
    datasets = {
        "mnist": MNIST,
        "fmnist": FashionMNIST,
        "kddcup99": KDD99,
        "amazon": Amazon,
        "vggface": VGGFace,
    }
    if (chosen_set := datasets.get(options.dataset)) is None:
        raise errors.MisconfigurationError(
            f"Dataset '{options.dataset}' does not exist, " +
            f"possible options: {set(datasets.keys())}"
        )
    data = chosen_set(
        f"./data/{options.dataset}",
        train=train,
        download=True,
        classes=classes
    )
    x_dim, y_dim = data.get_dims()
    return {
        "dataloader": torch.utils.data.DataLoader(
            data,
            batch_size=options.model_params['batch_size'],
            shuffle=shuffle,
            pin_memory=True,
        ),
        "x_dim": x_dim,
        "y_dim": y_dim,
    }
