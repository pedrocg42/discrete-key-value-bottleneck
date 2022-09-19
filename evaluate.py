import csv
import os

import fire
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import config
from data import CIFAR100
from utils_experiment import parse_experiment

DEVICE = "cuda"


@parse_experiment
def train(
    architecture: nn.Module,
    batch_size: int,
    device: str = DEVICE,
    **experiment,
):
    print(f"Evaluating experiment {experiment['name']}")

    # Building model
    model = architecture(**experiment)
    model.to(device)
    print(model)
    print(f"Encoder total parameters: {sum(param.numel() for param in model.encoder.parameters())}")
    print(f"Decoder total parameters: {sum(param.numel() for param in model.decoder.parameters())}")

    model.load_state_dict(torch.load(os.path.join(config.models_path, f"{experiment['name']}.pt")))

    # Preparing dataset
    test_dataset = CIFAR100(split="test")
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

    model.eval()

    # Extracting predictions and labels
    print(f" > Starting extraction of predictions and labels...")
    predictions = []
    labels = []
    pbar = tqdm(test_dataloader)
    for i, batch in enumerate(pbar):

        images, batch_labels = batch
        images = images.to(device)

        # Inference
        preds = model(images)

        # Saving results
        predictions.append(torch.argmax(preds.detach(), axis=1).cpu().numpy())
        labels.append(torch.argmax(batch_labels.detach(), axis=1).cpu().numpy())

    predictions = np.array(predictions).flatten()
    labels = np.array(labels).flatten()

    # Extracting metrics
    print(f" > Starting calculation of metrics...")
    csv_path = "results.csv"
    existed = os.path.exists(csv_path)

    f = open(csv_path, "a")
    csv_writer = csv.writer(f, delimiter="\t")

    if not existed:
        # Creating and writing header
        header = ["Experiment", "Accuracy"]
        for i in range(100):
            header.append(f"Acuraccy Class {i}")
        csv_writer.writerow(header)

    # Evaluating overall and for every single class
    row_results = [experiment["name"]]
    row_results.append(accuracy_score(predictions, labels))

    for i in range(100):
        indexes_class = np.where(labels == i)
        row_results.append(accuracy_score(predictions[indexes_class], labels[indexes_class]))

    csv_writer.writerow(row_results)
    print(f" > Results saved in {csv_path}")

    # close the file
    f.close()
    print("End")


if __name__ == "__main__":
    fire.Fire(train)
