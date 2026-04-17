import numpy as np
import torch 
import torch.nn as nn
from pathlib import Path

from toy_disentanglement.utils import get_activation_cls

ROOT = Path(__file__).parent.parent


# ==== Datasets ==== #

class LatentClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, latent_dim, num_samples, num_tasks, embedding_fn, bias=False):
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.embedding_fn = embedding_fn
        self.num_tasks = num_tasks
        self.bias = bias

        self.data_dist = torch.distributions.normal.Normal(
            torch.zeros(latent_dim), torch.ones(latent_dim)
        )

        self._init_data()

    @torch.no_grad()
    def _init_data(self):
        self.latents = self.data_dist.sample((self.num_samples,))  # (N, L)
        self.representations = self.embedding_fn(self.latents)  # (N, R)
        self.task_weights = torch.randn(self.latents.shape[1], self.num_tasks)
        self.task_weights = self.task_weights / torch.norm(self.task_weights, dim=0, keepdim=True)  # Normalize to unit length
        if self.bias:
            self.task_bias = torch.randn(self.num_tasks)
        else:
            self.task_bias = torch.zeros(self.num_tasks)
        self.task_labels = torch.sign(self.latents @ self.task_weights + self.task_bias)  # (N, T)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.latents[idx], self.representations[idx], self.task_labels[idx]


class LatentMLPDataset(torch.utils.data.Dataset):
    def __init__(self, latent_dim, num_samples, num_tasks, embedding_fn, sparsity=0.0, mlp_nonlinearity="relu", mlp_hidden_dim=64):
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.num_tasks = num_tasks
        self.embedding_fn = embedding_fn
        self.sparsity = sparsity
        self.mlp_nonlinearity = mlp_nonlinearity
        self.mlp_hidden_dim = mlp_hidden_dim

        self.data_dist = torch.distributions.normal.Normal(
            torch.zeros(latent_dim), torch.ones(latent_dim)
        )

        self._init_data()

    @torch.no_grad()
    def _init_data(self):
        self.latents = self.data_dist.sample((self.num_samples,))  # (N, L)
        self.representations = self.embedding_fn(self.latents)  # (N, R)
        self.jacobian_mask = torch.rand((self.num_tasks, self.latent_dim)) > self.sparsity  # (T, L)

        activation_cls = get_activation_cls(self.mlp_nonlinearity)
        mlps = []
        for t in range(self.num_tasks):
            mlp = nn.Sequential(
                nn.Linear(self.latent_dim, self.mlp_hidden_dim),
                activation_cls(),
                nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
                activation_cls(),
                nn.Linear(self.mlp_hidden_dim, 1),
            )
            mlp.layers[0].weight.data *= self.jacobian_mask[t].unsqueeze(1)  # Mask the input weights to enforce sparsity
            mlps.append(mlp)
        self.mlps = nn.ModuleList(mlps)

        self.task_targets = torch.cat([
            mlp(self.latents) for t, mlp in enumerate(self.mlps)
        ], dim=-1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.latents[idx], self.representations[idx], self.task_targets[idx]


# ==== Embedding network ==== #
    
def create_embedding_autoencoder(
    # autoencoder parameters
    input_dim: int = 5,
    representation_dim: int = 128,
    encoder_hidden_dims: list = [128, 128],
    decoder_hidden_dims: list = [128, 128],
    noise_std: float = 0.01,
    activation: str = "relu",
    # training parameters
    train: bool = True,  # optional
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    l2_penalty_weight: float = 0.1,
    pr_penalty_weight: float = 0.1,
    verbose: bool = False,
    # load checkpoint option
    checkpoint_path: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingAutoencoder(
        input_dim=input_dim,
        representation_dim=representation_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        noise_std=noise_std,
        activation=activation,
    ).to(device)

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(ROOT / checkpoint_path, map_location=device))
        return model

    data_dist = torch.distributions.normal.Normal(
        torch.zeros(input_dim), torch.ones(input_dim)
    )

    if train:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        for epoch in range(num_epochs):
            model.train()

            data = data_dist.sample((batch_size,))
            data = data.to(device)

            optimizer.zero_grad()

            x_hat, z = model(data)
            loss = criterion(x_hat, data)
            if l2_penalty_weight > 0:
                l2_penalty = torch.square(z).sum(dim=-1).mean(dim=0)
                loss += l2_penalty_weight * l2_penalty
            if pr_penalty_weight > 0:
                C = torch.cov(z.T)
                numerator = torch.trace(C) ** 2
                denominator = torch.sum(C ** 2)  # equivalent to tr(C²)
                pr = numerator / denominator
                loss -= pr_penalty_weight * pr  # want to maximized participation ratio

            loss.backward()
            optimizer.step()

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model


class GaussianNoise(nn.Module):
    def __init__(self, std):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        return x + torch.randn_like(x) * self.std


class EmbeddingEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[], noise_std=0.01, activation="relu"):
        super(EmbeddingEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.noise_std = noise_std
        self.activation = activation

        activation_cls = get_activation_cls(activation)

        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(GaussianNoise(noise_std))
            if i < len(dims) - 2:
                layers.append(activation_cls())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EmbeddingDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[], activation="relu"):
        super(EmbeddingDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation

        activation_cls = get_activation_cls(activation)

        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation_cls())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EmbeddingAutoencoder(nn.Module):
    def __init__(self, input_dim, representation_dim, encoder_hidden_dims=[], decoder_hidden_dims=[], noise_std=0.01, activation="relu"):
        super(EmbeddingAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims
        self.noise_std = noise_std
        self.activation = activation

        self.encoder = EmbeddingEncoder(input_dim, representation_dim, encoder_hidden_dims, noise_std, activation)
        self.decoder = EmbeddingDecoder(representation_dim, input_dim, decoder_hidden_dims, activation)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z