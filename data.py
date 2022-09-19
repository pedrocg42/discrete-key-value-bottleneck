import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import config


def unpickle(file):

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


class CIFAR100(Dataset):
    def __init__(self, train_val_split: float = 0.8, split: str = "train"):

        self.train_val_split = train_val_split
        self.split = split

        # Loading raw file
        if self.split == "train" or self.split == "val":
            file_path = os.path.join(config.dataset_path, "train")
        elif self.split == "test":
            file_path = os.path.join(config.dataset_path, "test")
        with open(file_path, "rb") as fo:
            data_dict = pickle.load(fo, encoding="bytes")

        # Loading images and labels
        self.images = data_dict[b"data"]
        self.labels = np.array(data_dict[b"fine_labels"])

        # Splitting data
        if self.split == "train" or self.split == "val":
            # Balancing and classes per split
            self.indexes = np.empty((100, 500), dtype=int)
            for i in range(100):
                self.indexes[i] = np.where(self.labels == i)[0]

            index_split = int(train_val_split * self.indexes.shape[1])
            if self.split == "train":
                self.indexes = self.indexes[:, :index_split].flatten()
            elif self.split == "val":
                self.indexes = self.indexes[:, index_split:].flatten()

            self.images = self.images[self.indexes]
            self.labels = self.labels[self.indexes]

        # Preprocessing images
        self.images = self.images.reshape(self.images.shape[0], 3, 32, 32)
        self.images = self.images / 255.0
        self.images = self.images.astype("float32")
        self.transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.images = self.transform(torch.from_numpy(self.images))

        # Loading and preprocessing labels
        self.labels = torch.LongTensor(self.labels)
        self.labels = F.one_hot(self.labels, num_classes=100).type(torch.float32)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class CIFAR100Exp1(Dataset):
    def __init__(self, class_num: int, train_val_split: float, steps: int, **kwargs):

        self.train_val_split = train_val_split
        self.class_num = class_num
        self.steps = steps

        # Loading raw file
        file_path = os.path.join(config.dataset_path, "train")
        with open(file_path, "rb") as fo:
            data_dict = pickle.load(fo, encoding="bytes")

        # Loading images and labels
        self.images = data_dict[b"data"]
        self.labels = np.array(data_dict[b"fine_labels"])

        # Splitting data
        # Balancing and classes per split
        self.indexes = np.empty((100, 500), dtype=int)
        for i in range(100):
            self.indexes[i] = np.where(self.labels == i)[0]

        index_split = int(train_val_split * self.indexes.shape[1])

        # Only taking val for one class
        self.indexes = self.indexes[self.class_num, index_split:]

        self.images = self.images[self.indexes]
        self.labels = self.labels[self.indexes]

        # Preprocessing images
        self.images = self.images.reshape(self.images.shape[0], 3, 32, 32)
        self.images = self.images / 255.0
        self.images = self.images.astype("float32")
        self.transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.images = self.transform(torch.from_numpy(self.images))

        # Loading and preprocessing labels
        self.labels = torch.LongTensor(self.labels)
        self.labels = F.one_hot(self.labels, num_classes=100).type(torch.float32)

        # Extracting dataset length
        self.length = self.images.shape[0]

    def __len__(self):
        return self.length * self.steps

    def __getitem__(self, index):
        return self.images[index % self.length], self.labels[index % self.length]


class CIFAR100Exp2(Dataset):
    def __init__(self, train_val_split: float, images_per_class: int, steps: int, **kwargs):

        self.train_val_split = train_val_split
        self.images_per_class = images_per_class
        self.steps = steps

        # Loading raw file
        file_path = os.path.join(config.dataset_path, "train")
        with open(file_path, "rb") as fo:
            data_dict = pickle.load(fo, encoding="bytes")

        # Loading images and labels
        self.images = data_dict[b"data"]
        self.labels = np.array(data_dict[b"fine_labels"])

        # Splitting data
        # Balancing and classes per split
        self.indexes = np.empty((100, 500), dtype=int)
        for i in range(100):
            self.indexes[i] = np.where(self.labels == i)[0]

        index_split = int(train_val_split * self.indexes.shape[1])

        # Only taking val
        self.indexes = self.indexes[:, index_split:]

        # Only taking some images per class
        self.indexes = self.indexes[:, : self.images_per_class].flatten()

        self.images = self.images[self.indexes]
        self.labels = self.labels[self.indexes]

        # Preprocessing images
        self.images = self.images.reshape(self.images.shape[0], 3, 32, 32)
        self.images = self.images / 255.0
        self.images = self.images.astype("float32")
        self.transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.images = self.transform(torch.from_numpy(self.images))

        # Loading and preprocessing labels
        self.labels = torch.LongTensor(self.labels)
        self.labels = F.one_hot(self.labels, num_classes=100).type(torch.float32)

        # Extracting dataset length
        self.length = self.images.shape[0]

    def __len__(self):
        return 100 * self.steps  # batch_size

    def __getitem__(self, index):
        return self.images[index % self.length], self.labels[index % self.length]


class CIFAR100Exp3(Dataset):
    def __init__(self, train_val_split: float, repetitions_per_class: int, steps: int, **kwargs):

        self.train_val_split = train_val_split
        self.repetitions_per_class = repetitions_per_class
        self.steps = steps

        # Loading raw file
        file_path = os.path.join(config.dataset_path, "train")
        with open(file_path, "rb") as fo:
            data_dict = pickle.load(fo, encoding="bytes")

        # Loading images and labels
        self.images = data_dict[b"data"]
        self.labels = np.array(data_dict[b"fine_labels"])

        # Splitting data
        # Balancing and classes per split
        self.indexes = np.empty((100, 500), dtype=int)
        for i in range(100):
            self.indexes[i] = np.where(self.labels == i)[0]

        index_split = int(train_val_split * self.indexes.shape[1])

        # Only taking val
        self.indexes = self.indexes[:, index_split:]

        # Repeating r times the indexes of each class
        self.indexes = np.tile(self.indexes, (1, self.repetitions_per_class))
        self.indexes = self.indexes.flatten()

        # Preprocessing images
        self.images = self.images.reshape(self.images.shape[0], 3, 32, 32)
        self.images = self.images / 255.0
        self.images = self.images.astype("float32")
        self.transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.images = self.transform(torch.from_numpy(self.images))

        # Loading and preprocessing labels
        self.labels = torch.LongTensor(self.labels)
        self.labels = F.one_hot(self.labels, num_classes=100).type(torch.float32)

        # Extracting dataset length (batch_size)
        self.length = self.images.shape[0]

    def __len__(self):
        return 100 * self.steps

    def __getitem__(self, index):
        i = self.indexes[index % len(self.indexes)]
        return self.images[i], self.labels[i]
