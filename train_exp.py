import os

import fire
import numpy as np
from vector_quantize_pytorch import VectorQuantize
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from tqdm import tqdm

import config
from utils_experiment import parse_experiment

DEVICE = "cuda"


@parse_experiment
def train(
    architecture: nn.Module,
    dataset: Dataset,
    train_val_split: float,
    batch_size: int,
    optimizer: nn.Module,
    criteria: nn.Module,
    learning_rate: float,
    steps: int,
    class_num: int = None,
    images_per_class: int = None,
    repetitions_per_class: int = None,
    shuffle: bool = True,
    device: str = DEVICE,
    **experiment,
):

    print(f"EXPERIMENT 1: Training small number of additional examples from a single class")

    # Building model
    model = architecture(**experiment)
    model.to(device)
    print(f"Encoder total parameters: {sum(param.numel() for param in model.encoder.parameters())}")
    print(f"Decoder total parameters: {sum(param.numel() for param in model.decoder.parameters())}")
    print(model)

    # Loading weights
    pretrained_model_name = experiment["name"].split("_exp")[0] + "_pretrain"
    model.load_state_dict(torch.load(os.path.join(config.models_path, f"{pretrained_model_name}.pt")))

    # Preparing dataset
    train_dataset = dataset(
        train_val_split=train_val_split,
        class_num=class_num,
        steps=steps,
        images_per_class=images_per_class,
        repetitions_per_class=repetitions_per_class,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True
    )

    # Configure Optimizer and Loss Function
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    criteria = criteria()

    # Initializing metrics
    accuracy = Accuracy().to(device)

    # Configurating model to only train certaing parts of it
    if experiment["architecture_type"] == "discrete_key_value_bottleneck":
        # All model frozen but values
        model.train(False)
    elif experiment["architecture_type"] == "vector_quantized":
        # Freezing Keys and encoder
        model.train(False)
        model.decoder.train(True)
    elif experiment["architecture_type"] == "baseline":
        # Freezing Keys and encoder
        model.train(False)
        model.decoder.train(True)

    # Start Training
    running_loss = 0.0
    batch_accuracies = []
    pbar = tqdm(train_dataloader)
    for i, batch in enumerate(pbar):

        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        # Zero fradient before every batch
        optimizer.zero_grad()

        # Inference
        output = model(images)

        # Compute loss
        loss = criteria(output, labels)
        loss.backward()

        # Adjust weights
        optimizer.step()

        # Computing loss
        running_loss += loss.item()
        avg_loss = running_loss / (i + 1)  # add batch_size

        # Computing accuracy
        batch_accuracies.append(accuracy(output.detach(), torch.argmax(labels.detach(), axis=1)).item())
        avg_accuracy = np.mean(batch_accuracies)

        pbar.set_postfix({"train_loss": avg_loss, "train_accuracy": avg_accuracy})

    # Saving model
    model_file_path = os.path.join(config.models_path, f"{experiment['name']}.pt")
    print(f" > Saving model in {model_file_path}")
    torch.save(model.state_dict(), model_file_path)

    print("End")


if __name__ == "__main__":
    fire.Fire(train)
