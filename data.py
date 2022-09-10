import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import config


def unpickle(file):

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


class CIFAR100(Dataset):
    def __init__(self, train_val_split: float = 0.8, split: str = "train"):

        # Loading raw file
        file_path = os.path.join(config.dataset_path, "train")
        with open(file_path, "rb") as fo:
            data_dict = pickle.load(fo, encoding="bytes")

        # Loading and preprocessing images
        self.images = data_dict[b"data"]
        self.images = self.images.reshape(self.images.shape[0], 3, 32, 32)
        self.images = self.images / 255.0
        self.images = self.images.astype("float32")
        self.transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.images = self.transform(torch.from_numpy(self.images))

        # Loading and preprocessing labels
        self.labels = torch.LongTensor(data_dict[b"fine_labels"])
        self.labels = F.one_hot(self.labels, num_classes=100)

        # Splitting data
        index_split = int(train_val_split * self.images.shape[0])
        if split == "train":
            self.images = self.images[:index_split]
            self.labels = self.labels[:index_split]
        elif split == "dev":
            self.images = self.images[index_split:]
            self.labels = self.labels[index_split:]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
