import os

import fire
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import config
from utils_experiment import parse_experiment


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
    **experiment,
):
    # Building model
    model = architecture(**experiment)
    model.cuda()
    print(model)
    print(f"Encoder total parameters: {sum(param.numel() for param in model.encoder.parameters())}")
    # print(f"Key-Value Bottleneck total parameters: {sum(param.numel() for param in model.encoder.parameters())}")
    print(f"Decoder total parameters: {sum(param.numel() for param in model.decoder.parameters())}")

    # Preparing dataset
    train_dataset = dataset(train_val_split=train_val_split)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    # Configure Optimizer
    optimizer = optimizer(model.parameters(), lr=learning_rate)

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
                images = images.cuda()
                output = model(images)

    print("[PHASE-1] Training Decoder and Values:")
    if experiment["architecture_type"] == "discrete_key_value_bottleneck":
        # Freezing Vector Quantizer
        model.key_value_bottleneck.vq.training = False
        for param in model.key_value_bottleneck.vq.parameters():
            param.requires_grad = False

    # Start Training
    for epoch in range(num_epochs):

        print(f" > Training epoch {epoch + 1} of {num_epochs_initialization_keys}")

        # Epoch training
        running_loss = 0.0
        pbar = tqdm(train_dataloader)
        for i, batch in enumerate(pbar):

            images, labels = batch

            # Zero fradient before every batch
            optimizer.zero_grad()

            # Inference
            images = images.cuda()
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
