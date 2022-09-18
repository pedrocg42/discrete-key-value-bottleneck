import os

import fire
import numpy as np
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
    num_epochs: int,
    num_epochs_initialization_keys: int = 10,
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
    val_dataset = dataset(train_val_split=train_val_split, split="val")
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    # Configure Optimizer and Loss Function
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    criteria = criteria()

    # Initializing Tensorboard logging and adding model graph
    writer = SummaryWriter(log_dir=os.path.join(config.logs_path, experiment["name"]))
    writer.add_graph(model, torch.zeros((100, 3, 32, 32), dtype=torch.float32).to(device))

    # Initializing metrics
    accuracy = Accuracy().to(device)

    model.train(True)

    if experiment["architecture_type"] in ["discrete_key_value_bottleneck", "vector_quantized"]:

        print("[PHASE-0] Keys Initialization:")
        # Start Training
        for epoch in range(num_epochs_initialization_keys):

            print(f" > Training epoch {epoch + 1} of {num_epochs_initialization_keys}")

            # Epoch training
            pbar = tqdm(train_dataloader)
            for i, batch in enumerate(pbar):

                images, labels = batch
                images = images.to(device)

                # Inference
                output = model(images)

    print("[PHASE-1] Training Classifier:")

    # Start Training
    for epoch in range(num_epochs):

        model.train(True)
        if experiment["architecture_type"] == "discrete_key_value_bottleneck":
            # Freezing Keys
            model.key_value_bottleneck.vq.train(False)
        elif experiment["architecture_type"] == "vector_quantized":
            # Freezing Keys
            model.vector_quantizer.train(False)

        print(f" > Training epoch {epoch + 1} of {num_epochs}")

        # Epoch training
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

        # Adding logs for every epoch
        writer.add_scalar("Loss/Train Loss", avg_loss, epoch)
        writer.add_scalar("Accuracy/Train Accuracy", avg_accuracy, epoch)

        # Epoch validation
        model.train(False)

        running_loss = 0.0
        batch_accuracies = []
        pbar = tqdm(val_dataloader)
        for i, batch in enumerate(pbar):

            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            # Inference
            output = model(images)

            # Compute loss
            loss = criteria(output, labels)

            # Computing loss
            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)  # add batch_size

            # Computing accuracy
            batch_accuracies.append(accuracy(output.detach(), torch.argmax(labels.detach(), axis=1)).item())
            avg_accuracy = np.mean(batch_accuracies)

            pbar.set_postfix({"val_loss": avg_loss, "val_accuracy": avg_accuracy})

        # Adding logs for every epoch
        writer.add_scalar("Loss/Validation Loss", avg_loss, epoch)
        writer.add_scalar("Accuracy/Validation Accuracy", avg_accuracy, epoch)

    # Saving model
    model_file_path = os.path.join(config.models_path, f"{experiment['name']}.pt")
    print(f" > Saving model in {model_file_path}")
    torch.save(model.state_dict(), model_file_path)

    del model
    writer.close()
    print("End")


if __name__ == "__main__":
    fire.Fire(train)
