import os

import fire
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

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
    num_epochs_initialization_keys: int,
    num_epochs: int,
    device: str = DEVICE,
    **experiment,
):
    # Building model
    model = architecture(**experiment)
    model.to(device)
    print(model)
    print(f"Encoder total parameters: {sum(param.numel() for param in model.encoder.parameters())}")
    # print(f"Key-Value Bottleneck total parameters: {sum(param.numel() for param in model.encoder.parameters())}")
    print(f"Decoder total parameters: {sum(param.numel() for param in model.decoder.parameters())}")

    # Preparing dataset
    train_dataset = dataset(train_val_split=train_val_split)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    # Configure Optimizer and Loss Function
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    criteria = criteria()

    # Initializing Tensorboard logging
    writer = SummaryWriter(log_dir=os.path.join(config.logs_path, experiment["name"]))

    # Creating output folder for saving the model
    output_folder = os.path.join(config.models_path, experiment["name"])
    os.makedirs(output_folder, exist_ok=True)

    model.train(True)

    if experiment["architecture_type"] == "discrete_key_value_bottleneck":

        print("[PHASE-0] Keys Initialization:")
        # Start Training
        for epoch in range(num_epochs_initialization_keys):

            print(f" > Training epoch {epoch + 1} of {num_epochs_initialization_keys}")

            # Epoch training
            pbar = tqdm(train_dataloader)
            for i, batch in enumerate(pbar):

                images, labels = batch

                # Inference
                images = images.to(device)
                output = model(images)

    if experiment["architecture_type"] == "discrete_key_value_bottleneck":
        print("[PHASE-1] Training Decoder and Values:")
        # Freezing Keys
        model.key_value_bottleneck.vq.train(False)
    else:
        print("Training Network:")

    # Start Training
    for epoch in range(num_epochs):

        print(f" > Training epoch {epoch + 1} of {num_epochs}")

        # Epoch training
        running_loss = 0.0
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
            pbar.set_postfix({"train_loss": avg_loss})

    print("End")


if __name__ == "__main__":
    fire.Fire(train)
