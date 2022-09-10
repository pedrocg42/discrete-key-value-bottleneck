from typing import List, Union
import torch
import torch.nn as nn

from vector_quantize_pytorch import VectorQuantize
from discrete_key_value_bottleneck_pytorch import DiscreteKeyValueBottleneck


class DKVB(nn.Module):
    def __init__(
        self,
        architecture_type: str,
        encoder_name: str = "dino_resnet_50",
        embedding_dim: int = 1024,
        key_value_pairs: int = 8192,
        num_codebooks: int = 512,
        value_dimension: Union[int, str] = "same",
        vq_decay: float = 0.9,
        threshold_ema_dead_code: int = 10,
        p_dropout: float = 0.2,
        hidden_dense_layers: List[int] = [512, 256],
        num_classes: int = 100,
        **kwargs,
    ):
        super(DKVB, self).__init__()

        self.architecture_type = architecture_type
        self.encoder_name = encoder_name
        self.embedding_dim = embedding_dim
        self.key_value_pairs = key_value_pairs
        self.num_codebooks = num_codebooks
        self.value_dimension = value_dimension
        self.vq_decay = vq_decay
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.p_dropout = p_dropout
        self.hidden_dense_layers = hidden_dense_layers
        self.num_classes = num_classes

        # embedding dimension and number of key-value pairs must be divisible by number of codes
        assert (self.embedding_dim % num_codebooks) == 0
        assert (self.key_value_pairs & num_codebooks) == 0

        if encoder_name == "dino_resnet_50":
            encoder_raw = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")

        # Getting all the model up to the second to last bottneck block
        self.encoder = nn.Sequential(
            encoder_raw.conv1,
            encoder_raw.bn1,
            encoder_raw.relu,
            encoder_raw.maxpool,
            encoder_raw.layer1,
            encoder_raw.layer2,
            encoder_raw.layer3,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        # Freezing the model
        for param in self.encoder.parameters():
            param.requires_grad = False

        if self.architecture_type == "discrete_key_value_bottleneck":
            if isinstance(self.value_dimension, str):
                self.value_dimension = self.embedding_dim // self.num_codebooks
            self.key_value_bottleneck = DiscreteKeyValueBottleneck(
                dim=self.embedding_dim,  # input dimension
                num_memory_codebooks=self.num_codebooks,  # number of memory codebook
                num_memories=self.embedding_dim // self.num_codebooks,  # number of memories
                dim_memory=self.embedding_dim // self.num_codebooks,  # dimension of the output memories
                decay=self.vq_decay,  # the exponential moving average decay, lower means the keys will change faster
                threshold_ema_dead_code=self.threshold_ema_dead_code,  # (0.8·batch-size·h·w·mz/num-pairs)
            )

        elif self.architecture_type == "vector_quantized":
            self.vector_quantizer = VectorQuantize(
                dim=self.embedding_dim,
                codebook_size=self.embedding_dim // self.num_codebooks,
                heads=self.num_codebooks,
                separate_codebook_per_head=True,
                decay=self.vq_decay,
                threshold_ema_dead_code=self.threshold_ema_dead_code,  # (0.8·batch-size·h·w·mz/num-pairs)
            )

        # Dense classification head
        decoder_module_list = nn.ModuleList()
        decoder_module_list.append(nn.Dropout(p=0.2))
        decoder_module_list.append(nn.Linear(self.embedding_dim, self.hidden_dense_layers[0]))
        for i in range(len(self.hidden_dense_layers) - 1):
            decoder_module_list.append(nn.Linear(self.hidden_dense_layers[i], self.hidden_dense_layers[i + 1]))
        decoder_module_list.append(nn.Linear(self.hidden_dense_layers[-1], num_classes))
        decoder_module_list.append(nn.Softmax(dim=1))

        self.decoder = nn.Sequential(*decoder_module_list)

    def forward(self, input):

        with torch.no_grad():
            embeddings = self.encoder(input)
            embeddings.detach_()

        if self.architecture_type == "discrete_key_value_bottleneck":
            # Reshaping embeddings to necessary format (batch, sequence, memory dimension)
            batch_size = input.shape[0]
            embeddings = torch.reshape(embeddings, (batch_size, 1, -1))
            # Creating memories
            memories = self.key_value_bottleneck(embeddings)
            # Processing final output
            output = self.decoder(torch.squeeze(memories))

        elif self.architecture_type == "vector_quantized":
            # Reshaping embeddings to necessary format (batch, sequence, memory dimension)
            batch_size = input.shape[0]
            embeddings = torch.reshape(embeddings, (batch_size, 1, -1))
            # Creating memories
            memories = self.vector_quantizer(embeddings)
            # Processing final output
            output = self.decoder(torch.squeeze(memories))

        elif self.architecture_type == "baseline":  # baseline classifier
            output = self.decoder(torch.squeeze(embeddings))

        return output
