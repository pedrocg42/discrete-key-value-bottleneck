# Architectures
from networks.discrete_key_value_bottleneck import DKVB

# Pretrained Encoders
from torchvision.models import resnet50

# Optimizers
from torch.optim import Adam

# Losses
from torch.nn import CrossEntropyLoss

# Datasets
from data import CIFAR100, CIFAR100Exp1Exp3

baseline_resnet50_pretrain = {
    "name": "baseline_resnet50_pretrain",
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
    "batch_size": 512,
    "num_epochs": 100,
}
baseline_resnet50_pretrain_encoder = baseline_resnet50_pretrain.copy()
baseline_resnet50_pretrain_encoder.update({"name": "baseline_resnet50_pretrain_encoder", "freeze_encoder": False})

dkvb_resnet50_1_pretrain = {
    "name": "dkvb_resnet50_1_pretrain",
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
    "batch_size": 512,
    "num_epochs_initialization_keys": 10,
    "num_epochs": 100,
}

dkvb_resnet50_64_pretrain = dkvb_resnet50_1_pretrain.copy()
dkvb_resnet50_64_pretrain.update({"name": "dkvb_resnet50_64_pretrain", "num_codebooks": 64})

dkvb_resnet50_128_pretrain = dkvb_resnet50_1_pretrain.copy()
dkvb_resnet50_128_pretrain.update({"name": "dkvb_resnet50_128_pretrain", "num_codebooks": 128})

dkvb_resnet50_512_pretrain = dkvb_resnet50_1_pretrain.copy()
dkvb_resnet50_512_pretrain.update({"name": "dkvb_resnet50_512_pretrain", "num_codebooks": 512})

vq_resnet50_1_pretrain = {
    "name": "vq_resnet50_1_pretrain",
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
    "batch_size": 512,
    "num_epochs": 100,
}


vq_resnet50_64_pretrain = vq_resnet50_1_pretrain.copy()
vq_resnet50_64_pretrain.update({"name": "vq_resnet50_64_pretrain", "num_codebooks": 64})

vq_resnet50_128_pretrain = vq_resnet50_1_pretrain.copy()
vq_resnet50_128_pretrain.update({"name": "vq_resnet50_128_pretrain", "num_codebooks": 128})

vq_resnet50_512_pretrain = vq_resnet50_1_pretrain.copy()
vq_resnet50_512_pretrain.update({"name": "vq_resnet50_512_pretrain", "num_codebooks": 512})


###############################################################
#########                 EXPERIMENT 1              ###########
###############################################################

baseline_resnet50_exp1 = {
    "name": "baseline_resnet50_exp1",
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
    "batch_size": 512,
    "num_epochs": 100,
}
baseline_resnet50_exp1_encoder = baseline_resnet50_exp1.copy()
baseline_resnet50_exp1_encoder.update({"name": "baseline_resnet50_exp1_encoder", "freeze_encoder": False})

dkvb_resnet50_1_exp1 = {
    "name": "dkvb_resnet50_1_exp1",
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
    "batch_size": 512,
    "num_epochs_initialization_keys": 10,
    "num_epochs": 100,
}

dkvb_resnet50_64_exp1 = dkvb_resnet50_1_exp1.copy()
dkvb_resnet50_64_exp1.update({"name": "dkvb_resnet50_64_exp1", "num_codebooks": 64})

dkvb_resnet50_128_exp1 = dkvb_resnet50_1_exp1.copy()
dkvb_resnet50_128_exp1.update({"name": "dkvb_resnet50_128_exp1", "num_codebooks": 128})

dkvb_resnet50_512_exp1 = dkvb_resnet50_1_exp1.copy()
dkvb_resnet50_512_exp1.update({"name": "dkvb_resnet50_512_exp1", "num_codebooks": 512})

vq_resnet50_1_exp1 = {
    "name": "vq_resnet50_1_exp1",
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
    "dataset": CIFAR100Exp1Exp3,
    "train_val_split": 0.8,
    # Train
    "optimizer": Adam,
    "criteria": CrossEntropyLoss,
    "learning_rate": 3e-4,
    "batch_size": 100,
    "num_epochs": 100,
}


vq_resnet50_64_exp1 = vq_resnet50_1_exp1.copy()
vq_resnet50_64_exp1.update({"name": "vq_resnet50_64_exp1", "num_codebooks": 64})

vq_resnet50_128_exp1 = vq_resnet50_1_exp1.copy()
vq_resnet50_128_exp1.update({"name": "vq_resnet50_128_exp1", "num_codebooks": 128})

vq_resnet50_512_exp1 = vq_resnet50_1_exp1.copy()
vq_resnet50_512_exp1.update({"name": "vq_resnet50_512_exp1", "num_codebooks": 512})
