# Architectures
from networks.discrete_key_value_bottleneck import DKVB

# Pretrained Encoders
from torchvision.models import resnet50

# Optimizers
from torch.optim import Adam

# Losses
from torch.nn import CrossEntropyLoss

# Datasets
from data import CIFAR100, CIFAR100Exp1, CIFAR100Exp2, CIFAR100Exp3

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

baseline_resnet50_encoder_pretrain = baseline_resnet50_pretrain.copy()
baseline_resnet50_encoder_pretrain.update({"name": "baseline_resnet50_encoder_pretrain", "freeze_encoder": False})

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
    "num_epochs_initialization_keys": 10,
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


########################## CLASS 0 ############################

baseline_resnet50_exp1_class_0 = {
    "name": "baseline_resnet50_exp1_class_0",
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
    "dataset": CIFAR100Exp1,
    "train_val_split": 0.8,
    "class_num": 0,
    # Train
    "optimizer": Adam,
    "criteria": CrossEntropyLoss,
    "learning_rate": 3e-4,
    "batch_size": 100,
    "steps": 250,
}

baseline_resnet50_encoder_exp1_class_0 = baseline_resnet50_exp1_class_0.copy()
baseline_resnet50_encoder_exp1_class_0.update(
    {"name": "baseline_resnet50_exp1_class_0_encoder", "freeze_encoder": False}
)

dkvb_resnet50_1_exp1_class_0 = {
    "name": "dkvb_resnet50_1_exp1_class_0",
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
    "freeze_decoder": True,
    # Data
    "dataset": CIFAR100Exp1,
    "train_val_split": 0.8,
    "class_num": 0,
    # Train
    "optimizer": Adam,
    "criteria": CrossEntropyLoss,
    "learning_rate": 3e-4,
    "batch_size": 100,
    "steps": 250,
}

dkvb_resnet50_64_exp1_class_0 = dkvb_resnet50_1_exp1_class_0.copy()
dkvb_resnet50_64_exp1_class_0.update({"name": "dkvb_resnet50_64_exp1_class_0", "num_codebooks": 64})

dkvb_resnet50_128_exp1_class_0 = dkvb_resnet50_1_exp1_class_0.copy()
dkvb_resnet50_128_exp1_class_0.update({"name": "dkvb_resnet50_128_exp1_class_0", "num_codebooks": 128})

dkvb_resnet50_512_exp1_class_0 = dkvb_resnet50_1_exp1_class_0.copy()
dkvb_resnet50_512_exp1_class_0.update({"name": "dkvb_resnet50_512_exp1_class_0", "num_codebooks": 512})

vq_resnet50_1_exp1_class_0 = {
    "name": "vq_resnet50_1_exp1_class_0",
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
    "dataset": CIFAR100Exp1,
    "train_val_split": 0.8,
    "class_num": 0,
    # Train
    "optimizer": Adam,
    "criteria": CrossEntropyLoss,
    "learning_rate": 3e-4,
    "batch_size": 100,
    "steps": 250,
}


vq_resnet50_64_exp1_class_0 = vq_resnet50_1_exp1_class_0.copy()
vq_resnet50_64_exp1_class_0.update({"name": "vq_resnet50_64_exp1_class_0", "num_codebooks": 64})

vq_resnet50_128_exp1_class_0 = vq_resnet50_1_exp1_class_0.copy()
vq_resnet50_128_exp1_class_0.update({"name": "vq_resnet50_128_exp1_class_0", "num_codebooks": 128})

vq_resnet50_512_exp1_class_0 = vq_resnet50_1_exp1_class_0.copy()
vq_resnet50_512_exp1_class_0.update({"name": "vq_resnet50_512_exp1_class_0", "num_codebooks": 512})


########################## CLASS 28 ############################

baseline_resnet50_exp1_class_28 = baseline_resnet50_exp1_class_0.copy()
baseline_resnet50_exp1_class_28.update(
    {
        "name": "baseline_resnet50_exp1_class_28",
        "class_num": 28,
    }
)


baseline_resnet50_encoder_exp1_class_28 = baseline_resnet50_encoder_exp1_class_0.copy()
baseline_resnet50_encoder_exp1_class_28.update(
    {
        "name": "baseline_resnet50_encoder_exp1_class_28",
        "class_num": 28,
    }
)

dkvb_resnet50_1_exp1_class_28 = dkvb_resnet50_1_exp1_class_0.copy()
dkvb_resnet50_1_exp1_class_28.update(
    {
        "name": "dkvb_resnet50_1_exp1_class_28",
        "class_num": 28,
    }
)

dkvb_resnet50_64_exp1_class_28 = dkvb_resnet50_64_exp1_class_0.copy()
dkvb_resnet50_64_exp1_class_28.update(
    {
        "name": "dkvb_resnet50_64_exp1_class_28",
        "class_num": 28,
    }
)

dkvb_resnet50_128_exp1_class_28 = dkvb_resnet50_128_exp1_class_0.copy()
dkvb_resnet50_128_exp1_class_28.update(
    {
        "name": "dkvb_resnet50_128_exp1_class_28",
        "class_num": 28,
    }
)

dkvb_resnet50_512_exp1_class_28 = dkvb_resnet50_512_exp1_class_0.copy()
dkvb_resnet50_512_exp1_class_28.update(
    {
        "name": "dkvb_resnet50_512_exp1_class_28",
        "class_num": 28,
    }
)

vq_resnet50_1_exp1_class_28 = vq_resnet50_1_exp1_class_0.copy()
vq_resnet50_1_exp1_class_28.update(
    {
        "name": "vq_resnet50_1_exp1_class_28",
        "class_num": 28,
    }
)

vq_resnet50_64_exp1_class_28 = vq_resnet50_64_exp1_class_0.copy()
vq_resnet50_64_exp1_class_28.update(
    {
        "name": "vq_resnet50_64_exp1_class_28",
        "class_num": 28,
    }
)


vq_resnet50_128_exp1_class_28 = vq_resnet50_128_exp1_class_0.copy()
vq_resnet50_128_exp1_class_28.update(
    {
        "name": "vq_resnet50_128_exp1_class_28",
        "class_num": 28,
    }
)


vq_resnet50_512_exp1_class_28 = vq_resnet50_512_exp1_class_0.copy()
vq_resnet50_512_exp1_class_28.update(
    {
        "name": "vq_resnet50_512_exp1_class_28",
        "class_num": 28,
    }
)


########################## CLASS 99 ############################

baseline_resnet50_exp1_class_99 = baseline_resnet50_exp1_class_0.copy()
baseline_resnet50_exp1_class_99.update(
    {
        "name": "baseline_resnet50_exp1_class_99",
        "class_num": 99,
    }
)


baseline_resnet50_encoder_exp1_class_99 = baseline_resnet50_exp1_class_0.copy()
baseline_resnet50_encoder_exp1_class_99.update(
    {
        "name": "baseline_resnet50_encoder_exp1_class_99",
        "class_num": 99,
    }
)

dkvb_resnet50_1_exp1_class_99 = dkvb_resnet50_1_exp1_class_0.copy()
dkvb_resnet50_1_exp1_class_99.update(
    {
        "name": "dkvb_resnet50_1_exp1_class_99",
        "class_num": 99,
    }
)

dkvb_resnet50_64_exp1_class_99 = dkvb_resnet50_64_exp1_class_0.copy()
dkvb_resnet50_64_exp1_class_99.update(
    {
        "name": "dkvb_resnet50_64_exp1_class_99",
        "class_num": 99,
    }
)

dkvb_resnet50_128_exp1_class_99 = dkvb_resnet50_128_exp1_class_0.copy()
dkvb_resnet50_128_exp1_class_99.update(
    {
        "name": "dkvb_resnet50_128_exp1_class_99",
        "class_num": 99,
    }
)

dkvb_resnet50_512_exp1_class_99 = dkvb_resnet50_512_exp1_class_0.copy()
dkvb_resnet50_512_exp1_class_99.update(
    {
        "name": "dkvb_resnet50_512_exp1_class_99",
        "class_num": 99,
    }
)

vq_resnet50_1_exp1_class_99 = vq_resnet50_1_exp1_class_0.copy()
vq_resnet50_1_exp1_class_99.update(
    {
        "name": "vq_resnet50_1_exp1_class_99",
        "class_num": 99,
    }
)

vq_resnet50_64_exp1_class_99 = vq_resnet50_64_exp1_class_0.copy()
vq_resnet50_64_exp1_class_99.update(
    {
        "name": "vq_resnet50_64_exp1_class_99",
        "class_num": 99,
    }
)


vq_resnet50_128_exp1_class_99 = vq_resnet50_128_exp1_class_0.copy()
vq_resnet50_128_exp1_class_99.update(
    {
        "name": "vq_resnet50_128_exp1_class_99",
        "class_num": 99,
    }
)


vq_resnet50_512_exp1_class_99 = vq_resnet50_512_exp1_class_0.copy()
vq_resnet50_512_exp1_class_99.update(
    {
        "name": "vq_resnet50_512_exp1_class_99",
        "class_num": 99,
    }
)


###############################################################
#########                 EXPERIMENT 2              ###########
###############################################################


##################### Additional Samples 50 ###################

baseline_resnet50_exp2_samples_50 = {
    "name": "baseline_resnet50_exp2_samples_50",
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
    "dataset": CIFAR100Exp2,
    "train_val_split": 0.8,
    "images_per_class": 50,
    # Train
    "optimizer": Adam,
    "criteria": CrossEntropyLoss,
    "learning_rate": 3e-4,
    "batch_size": 100,
    "steps": 10000,
}

baseline_resnet50_encoder_exp2_samples_50 = baseline_resnet50_exp2_samples_50.copy()
baseline_resnet50_encoder_exp2_samples_50.update(
    {"name": "baseline_resnet50_encoder_exp2_samples_50", "freeze_encoder": False}
)

dkvb_resnet50_1_exp2_samples_50 = {
    "name": "dkvb_resnet50_1_exp2_samples_50",
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
    "freeze_decoder": True,
    # Data
    "dataset": CIFAR100Exp2,
    "train_val_split": 0.8,
    "images_per_class": 50,
    # Train
    "optimizer": Adam,
    "criteria": CrossEntropyLoss,
    "learning_rate": 3e-4,
    "batch_size": 100,
    "steps": 10000,
}

dkvb_resnet50_64_exp2_samples_50 = dkvb_resnet50_1_exp2_samples_50.copy()
dkvb_resnet50_64_exp2_samples_50.update({"name": "dkvb_resnet50_64_exp2_samples_50", "num_codebooks": 64})

dkvb_resnet50_128_exp2_samples_50 = dkvb_resnet50_1_exp2_samples_50.copy()
dkvb_resnet50_128_exp2_samples_50.update({"name": "dkvb_resnet50_128_exp2_samples_50", "num_codebooks": 128})

dkvb_resnet50_512_exp2_samples_50 = dkvb_resnet50_1_exp2_samples_50.copy()
dkvb_resnet50_512_exp2_samples_50.update({"name": "dkvb_resnet50_512_exp2_samples_50", "num_codebooks": 512})

vq_resnet50_1_exp2_samples_50 = {
    "name": "vq_resnet50_1_exp2_samples_50",
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
    "dataset": CIFAR100Exp2,
    "train_val_split": 0.8,
    "images_per_class": 50,
    # Train
    "optimizer": Adam,
    "criteria": CrossEntropyLoss,
    "learning_rate": 3e-4,
    "batch_size": 100,
    "steps": 10000,
}


vq_resnet50_64_exp2_samples_50 = vq_resnet50_1_exp2_samples_50.copy()
vq_resnet50_64_exp2_samples_50.update({"name": "vq_resnet50_64_exp2_samples_50", "num_codebooks": 64})

vq_resnet50_128_exp2_samples_50 = vq_resnet50_1_exp2_samples_50.copy()
vq_resnet50_128_exp2_samples_50.update({"name": "vq_resnet50_128_exp2_samples_50", "num_codebooks": 128})

vq_resnet50_512_exp2_samples_50 = vq_resnet50_1_exp2_samples_50.copy()
vq_resnet50_512_exp2_samples_50.update({"name": "vq_resnet50_512_exp2_samples_50", "num_codebooks": 512})


##################### Additional Samples 70 ###################

baseline_resnet50_exp2_samples_70 = baseline_resnet50_exp2_samples_50.copy()
baseline_resnet50_exp2_samples_70.update(
    {
        "name": "baseline_resnet50_exp2_samples_70",
        "images_per_class": 70,
    }
)


baseline_resnet50_encoder_exp2_samples_70 = baseline_resnet50_encoder_exp2_samples_50.copy()
baseline_resnet50_encoder_exp2_samples_70.update(
    {
        "name": "baseline_resnet50_encoder_exp2_samples_70",
        "images_per_class": 70,
    }
)

dkvb_resnet50_1_exp2_samples_70 = dkvb_resnet50_1_exp2_samples_50.copy()
dkvb_resnet50_1_exp2_samples_70.update(
    {
        "name": "dkvb_resnet50_1_exp2_samples_70",
        "images_per_class": 70,
    }
)

dkvb_resnet50_64_exp2_samples_70 = dkvb_resnet50_64_exp2_samples_50.copy()
dkvb_resnet50_64_exp2_samples_70.update(
    {
        "name": "dkvb_resnet50_64_exp2_samples_70",
        "images_per_class": 70,
    }
)

dkvb_resnet50_128_exp2_samples_70 = dkvb_resnet50_128_exp2_samples_50.copy()
dkvb_resnet50_128_exp2_samples_70.update(
    {
        "name": "dkvb_resnet50_128_exp2_samples_70",
        "images_per_class": 70,
    }
)

dkvb_resnet50_512_exp2_samples_70 = dkvb_resnet50_512_exp2_samples_50.copy()
dkvb_resnet50_512_exp2_samples_70.update(
    {
        "name": "dkvb_resnet50_512_exp2_samples_70",
        "images_per_class": 70,
    }
)

vq_resnet50_1_exp2_samples_70 = vq_resnet50_1_exp2_samples_50.copy()
vq_resnet50_1_exp2_samples_70.update(
    {
        "name": "vq_resnet50_1_exp2_samples_70",
        "images_per_class": 70,
    }
)

vq_resnet50_64_exp2_samples_70 = vq_resnet50_64_exp2_samples_50.copy()
vq_resnet50_64_exp2_samples_70.update(
    {
        "name": "vq_resnet50_64_exp2_samples_70",
        "images_per_class": 70,
    }
)


vq_resnet50_128_exp2_samples_70 = vq_resnet50_128_exp2_samples_50.copy()
vq_resnet50_128_exp2_samples_70.update(
    {
        "name": "vq_resnet50_128_exp2_samples_70",
        "images_per_class": 70,
    }
)


vq_resnet50_512_exp2_samples_70 = vq_resnet50_512_exp2_samples_50.copy()
vq_resnet50_512_exp2_samples_70.update(
    {
        "name": "vq_resnet50_512_exp2_samples_70",
        "images_per_class": 70,
    }
)

##################### Additional Samples 100 ###################

baseline_resnet50_exp2_samples_100 = baseline_resnet50_exp2_samples_50.copy()
baseline_resnet50_exp2_samples_100.update(
    {
        "name": "baseline_resnet50_exp2_samples_100",
        "images_per_class": 100,
    }
)


baseline_resnet50_encoder_exp2_samples_100 = baseline_resnet50_encoder_exp2_samples_50.copy()
baseline_resnet50_encoder_exp2_samples_100.update(
    {
        "name": "baseline_resnet50_encoder_exp2_samples_100",
        "images_per_class": 100,
    }
)

dkvb_resnet50_1_exp2_samples_100 = dkvb_resnet50_1_exp2_samples_50.copy()
dkvb_resnet50_1_exp2_samples_100.update(
    {
        "name": "dkvb_resnet50_1_exp2_samples_100",
        "images_per_class": 100,
    }
)

dkvb_resnet50_64_exp2_samples_100 = dkvb_resnet50_64_exp2_samples_50.copy()
dkvb_resnet50_64_exp2_samples_100.update(
    {
        "name": "dkvb_resnet50_64_exp2_samples_100",
        "images_per_class": 100,
    }
)

dkvb_resnet50_128_exp2_samples_100 = dkvb_resnet50_128_exp2_samples_50.copy()
dkvb_resnet50_128_exp2_samples_100.update(
    {
        "name": "dkvb_resnet50_128_exp2_samples_100",
        "images_per_class": 100,
    }
)

dkvb_resnet50_512_exp2_samples_100 = dkvb_resnet50_512_exp2_samples_50.copy()
dkvb_resnet50_512_exp2_samples_100.update(
    {
        "name": "dkvb_resnet50_512_exp2_samples_100",
        "images_per_class": 100,
    }
)

vq_resnet50_1_exp2_samples_100 = vq_resnet50_1_exp2_samples_50.copy()
vq_resnet50_1_exp2_samples_100.update(
    {
        "name": "vq_resnet50_1_exp2_samples_100",
        "images_per_class": 100,
    }
)

vq_resnet50_64_exp2_samples_100 = vq_resnet50_64_exp2_samples_50.copy()
vq_resnet50_64_exp2_samples_100.update(
    {
        "name": "vq_resnet50_64_exp2_samples_100",
        "images_per_class": 100,
    }
)


vq_resnet50_128_exp2_samples_100 = vq_resnet50_128_exp2_samples_50.copy()
vq_resnet50_128_exp2_samples_100.update(
    {
        "name": "vq_resnet50_128_exp2_samples_100",
        "images_per_class": 100,
    }
)


vq_resnet50_512_exp2_samples_100 = vq_resnet50_512_exp2_samples_50.copy()
vq_resnet50_512_exp2_samples_100.update(
    {
        "name": "vq_resnet50_512_exp2_samples_100",
        "images_per_class": 100,
    }
)

###############################################################
#########                 EXPERIMENT 3              ###########
###############################################################


######################## Repetititons 20 ######################

baseline_resnet50_exp3_reps_20 = {
    "name": "baseline_resnet50_exp3_reps_20",
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
    "dataset": CIFAR100Exp3,
    "train_val_split": 0.8,
    "repetitions_per_class": 20,
    "shuffle": False,
    # Train
    "optimizer": Adam,
    "criteria": CrossEntropyLoss,
    "learning_rate": 3e-4,
    "batch_size": 100,
    "steps": 10000,
}

baseline_resnet50_encoder_exp3_reps_20 = baseline_resnet50_exp3_reps_20.copy()
baseline_resnet50_encoder_exp3_reps_20.update(
    {"name": "baseline_resnet50_encoder_exp3_reps_20", "freeze_encoder": False}
)

dkvb_resnet50_1_exp3_reps_20 = {
    "name": "dkvb_resnet50_1_exp3_reps_20",
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
    "freeze_decoder": True,
    # Data
    "dataset": CIFAR100Exp3,
    "train_val_split": 0.8,
    "repetitions_per_class": 20,
    "shuffle": False,
    # Train
    "optimizer": Adam,
    "criteria": CrossEntropyLoss,
    "learning_rate": 3e-4,
    "batch_size": 100,
    "steps": 10000,
}

dkvb_resnet50_64_exp3_reps_20 = dkvb_resnet50_1_exp3_reps_20.copy()
dkvb_resnet50_64_exp3_reps_20.update({"name": "dkvb_resnet50_64_exp3_reps_20", "num_codebooks": 64})

dkvb_resnet50_128_exp3_reps_20 = dkvb_resnet50_1_exp3_reps_20.copy()
dkvb_resnet50_128_exp3_reps_20.update({"name": "dkvb_resnet50_128_exp3_reps_20", "num_codebooks": 128})

dkvb_resnet50_512_exp3_reps_20 = dkvb_resnet50_1_exp3_reps_20.copy()
dkvb_resnet50_512_exp3_reps_20.update({"name": "dkvb_resnet50_512_exp3_reps_20", "num_codebooks": 512})

vq_resnet50_1_exp3_reps_20 = {
    "name": "vq_resnet50_1_exp3_reps_20",
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
    "dataset": CIFAR100Exp3,
    "train_val_split": 0.8,
    "repetitions_per_class": 20,
    "shuffle": False,
    # Train
    "optimizer": Adam,
    "criteria": CrossEntropyLoss,
    "learning_rate": 3e-4,
    "batch_size": 100,
    "steps": 10000,
}


vq_resnet50_64_exp3_reps_20 = vq_resnet50_1_exp3_reps_20.copy()
vq_resnet50_64_exp3_reps_20.update({"name": "vq_resnet50_64_exp3_reps_20", "num_codebooks": 64})

vq_resnet50_128_exp3_reps_20 = vq_resnet50_1_exp3_reps_20.copy()
vq_resnet50_128_exp3_reps_20.update({"name": "vq_resnet50_128_exp3_reps_20", "num_codebooks": 128})

vq_resnet50_512_exp3_reps_20 = vq_resnet50_1_exp3_reps_20.copy()
vq_resnet50_512_exp3_reps_20.update({"name": "vq_resnet50_512_exp3_reps_20", "num_codebooks": 512})


##################### Repetitions 10 ###################

baseline_resnet50_exp3_reps_10 = baseline_resnet50_exp3_reps_20.copy()
baseline_resnet50_exp3_reps_10.update(
    {
        "name": "baseline_resnet50_exp3_reps_10",
        "repetitions_per_class": 10,
    }
)


baseline_resnet50_encoder_exp3_reps_10 = baseline_resnet50_encoder_exp3_reps_20.copy()
baseline_resnet50_encoder_exp3_reps_10.update(
    {
        "name": "baseline_resnet50_encoder_exp3_reps_10",
        "repetitions_per_class": 10,
    }
)

dkvb_resnet50_1_exp3_reps_10 = dkvb_resnet50_1_exp3_reps_20.copy()
dkvb_resnet50_1_exp3_reps_10.update(
    {
        "name": "dkvb_resnet50_1_exp3_reps_10",
        "repetitions_per_class": 10,
    }
)

dkvb_resnet50_64_exp3_reps_10 = dkvb_resnet50_64_exp3_reps_20.copy()
dkvb_resnet50_64_exp3_reps_10.update(
    {
        "name": "dkvb_resnet50_64_exp3_reps_10",
        "repetitions_per_class": 10,
    }
)

dkvb_resnet50_128_exp3_reps_10 = dkvb_resnet50_128_exp3_reps_20.copy()
dkvb_resnet50_128_exp3_reps_10.update(
    {
        "name": "dkvb_resnet50_128_exp3_reps_10",
        "repetitions_per_class": 10,
    }
)

dkvb_resnet50_512_exp3_reps_10 = dkvb_resnet50_512_exp3_reps_20.copy()
dkvb_resnet50_512_exp3_reps_10.update(
    {
        "name": "dkvb_resnet50_512_exp3_reps_10",
        "repetitions_per_class": 10,
    }
)

vq_resnet50_1_exp3_reps_10 = vq_resnet50_1_exp3_reps_20.copy()
vq_resnet50_1_exp3_reps_10.update(
    {
        "name": "vq_resnet50_1_exp3_reps_10",
        "repetitions_per_class": 10,
    }
)

vq_resnet50_64_exp3_reps_10 = vq_resnet50_64_exp3_reps_20.copy()
vq_resnet50_64_exp3_reps_10.update(
    {
        "name": "vq_resnet50_64_exp3_reps_10",
        "repetitions_per_class": 10,
    }
)


vq_resnet50_128_exp3_reps_10 = vq_resnet50_128_exp3_reps_20.copy()
vq_resnet50_128_exp3_reps_10.update(
    {
        "name": "vq_resnet50_128_exp3_reps_10",
        "repetitions_per_class": 10,
    }
)


vq_resnet50_512_exp3_reps_10 = vq_resnet50_512_exp3_reps_20.copy()
vq_resnet50_512_exp3_reps_10.update(
    {
        "name": "vq_resnet50_512_exp3_reps_10",
        "repetitions_per_class": 10,
    }
)
