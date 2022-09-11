# Architectures
from networks.discrete_key_value_bottleneck import DKVB

# Pretrained Encoders
from torchvision.models import resnet18

# Optimizers
from torch.optim import Adam

# Losses
from torch.nn import CrossEntropyLoss

# Datasets
from data import CIFAR100

dkvb_resnet18_1_pretrain = {
    "name": "dkvb_resnet18_1_pretrain",
    # Model
    "architecture": DKVB,
    "architecture_type": "discrete_key_value_bottleneck",
    "encoder_name": "dino_resnet_50",
    "embedding_dim": 1024,
    "key_value_pairs": 8192,
    "num_codebooks": 1,
    "value_dimension": "same",
    "vq_decay": 0.9,
    "threshold_ema_dead_code": 10,  # 0.8 * 100 * 1 * 1 * 1024 / 8192 (0.8·batch-size·h·w·mz/num-pairs)
    "p_dropout": 0.2,
    "hidden_dense_layers": [512, 256],
    "num_classes": 100,
    # Data
    "dataset": CIFAR100,
    "train_val_split": 0.8,
    # Train
    "optimizer": Adam,
    "criteria": CrossEntropyLoss,
    "learning_rate": 3e-4,
    "batch_size": 100,
    "num_epochs_initialization_keys": 10,
    "num_epochs": 100,
}

vq_resnet18_1_pretrain = {
    "name": "vq_resnet18_1_pretrain",
    # Model
    "architecture": DKVB,
    "architecture_type": "vector_quantized",
    "encoder_name": "dino_resnet_50",
    "embedding_dim": 1024,
    "key_value_pairs": 8192,
    "num_codebooks": 1,
    "value_dimension": "same",
    "vq_decay": 0.9,
    "threshold_ema_dead_code": 10,  # 0.8 * 100 * 1 * 1 * 1024 / 8192 (0.8·batch-size·h·w·mz/num-pairs)
    "p_dropout": 0.2,
    "hidden_dense_layers": [512, 256],
    "num_classes": 100,
    # Data
    "dataset": CIFAR100,
    "train_val_split": 0.8,
    # Train
    "optimizer": Adam,
    "criteria": CrossEntropyLoss,
    "learning_rate": 3e-4,
    "batch_size": 100,
    "num_epochs": 100,
}

baseline_resnet18_pretrain = {
    "name": "baseline_resnet18_pretrain",
    # Model
    "architecture": DKVB,
    "architecture_type": "baseline",
    "encoder_name": "dino_resnet_50",
    "embedding_dim": 1024,
    "key_value_pairs": 8192,
    "num_codebooks": 1,
    "value_dimension": "same",
    "vq_decay": 0.9,
    "threshold_ema_dead_code": 10,  # 0.8 * 100 * 1 * 1 * 1024 / 8192 (0.8·batch-size·h·w·mz/num-pairs)
    "p_dropout": 0.2,
    "hidden_dense_layers": [512, 256],
    "num_classes": 100,
    # Data
    "dataset": CIFAR100,
    "train_val_split": 0.8,
    # Train
    "optimizer": Adam,
    "criteria": CrossEntropyLoss,
    "learning_rate": 3e-4,
    "batch_size": 100,
    "num_epochs": 100,
}
