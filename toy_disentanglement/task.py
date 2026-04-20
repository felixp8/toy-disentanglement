import numpy as np
import torch 
import torch.nn as nn
import warnings
from pathlib import Path

from toy_disentanglement.utils import get_activation_cls

ROOT = Path(__file__).parent.parent


# ==== Datasets ==== #

class LatentSparseLinearDataset(torch.utils.data.Dataset):
    def __init__(self, latent_dim, num_samples, num_tasks, embedding_fn, bias=False, correlation=0.0, sparsity=0.0, task_correlation=0.0):
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.embedding_fn = embedding_fn
        self.num_tasks = num_tasks
        self.bias = bias
        self.correlation = correlation
        self.task_correlation = task_correlation

        if (1 - sparsity) < (1 / latent_dim):
            warnings.warn(
                f"Sparsity level {sparsity} is too high for latent dimension {latent_dim}, "
                f"must have at least 1 nonzero weight per task. Setting sparsity to {1 - 1/latent_dim}."
            )
            sparsity = 1 - 1/latent_dim
        self.sparsity = sparsity

        if correlation != 0.0:
            covariance = torch.eye(latent_dim) * (1 - correlation) + torch.ones((latent_dim, latent_dim)) * correlation
            self.data_dist = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(latent_dim), covariance
            )
        else:
            self.data_dist = torch.distributions.normal.Normal(
                torch.zeros(latent_dim), torch.ones(latent_dim)
            )

        self._init_data()

    @torch.no_grad()
    def _init_data(self):
        self.latents = self.data_dist.sample((self.num_samples,))  # (N, L)
        self.representations = self.embedding_fn(self.latents)  # (N, R)

        if self.task_correlation != 0.0:
            task_covariance = torch.eye(self.latent_dim) * (1 - self.task_correlation) + torch.ones((self.latent_dim, self.latent_dim)) * self.task_correlation
            task_weights_dist = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(self.latent_dim), task_covariance
            )
            self.task_weights = task_weights_dist.sample((self.num_tasks,)).T  # (L, T)
        else:
            self.task_weights = torch.randn(self.latents.shape[1], self.num_tasks)

        for t in range(self.num_tasks):
            if (1 - self.sparsity) <= (1 / self.latent_dim):
                num_nonzero = 1  # Ensure at least 1 nonzero weight
            elif self.sparsity == 0.0:
                num_nonzero = self.latent_dim  # No sparsity, all weights are nonzero
            else:
                num_nonzero = 1 + torch.binomial(
                    torch.tensor(self.latent_dim - 1.0),
                    torch.tensor((self.latent_dim * (1 - self.sparsity) - 1) / (self.latent_dim - 1)),
                ).int().item()
            # nonzero_indices = torch.randperm(self.latent_dim)[:num_nonzero]
            zero_indices = torch.randperm(self.latent_dim)[:(self.latent_dim - num_nonzero)]
            self.task_weights[zero_indices, t] = 0.0
        self.task_weights = self.task_weights / torch.norm(self.task_weights, dim=0, keepdim=True)  # Normalize to unit length

        if self.bias:
            self.task_bias = torch.rand(self.num_tasks) * 2 - 1  # uniform over [-1, 1]
        else:
            self.task_bias = torch.zeros(self.num_tasks)

        self._generate_labels()
    
    @torch.no_grad()
    def _generate_labels(self):
        self.task_labels = self.latents @ self.task_weights + self.task_bias  # (N, T)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.latents[idx], self.representations[idx], self.task_labels[idx]
    

class LatentClassificationDataset(LatentSparseLinearDataset):
    def __init__(self, latent_dim, num_samples, num_tasks, embedding_fn, bias=False, correlation=0.0, sparsity=0.0, task_correlation=0.0):
        super().__init__(
            latent_dim=latent_dim, num_samples=num_samples, num_tasks=num_tasks, 
            embedding_fn=embedding_fn, bias=bias, correlation=correlation, sparsity=sparsity, task_correlation=task_correlation
        )

    @torch.no_grad()
    def _generate_labels(self):
        self.task_labels = torch.sign(self.latents @ self.task_weights + self.task_bias)  # (N, T)


class LatentTanhDataset(LatentSparseLinearDataset):
    def __init__(self, latent_dim, num_samples, num_tasks, embedding_fn, bias=False, correlation=0.0, sparsity=0.0, task_correlation=0.0, tanh_scale=1.0):
        assert tanh_scale > 0, "tanh_scale must be positive"
        self.tanh_scale = tanh_scale
        super().__init__(
            latent_dim=latent_dim, num_samples=num_samples, num_tasks=num_tasks, 
            embedding_fn=embedding_fn, bias=bias, correlation=correlation, sparsity=sparsity, task_correlation=task_correlation
        )

    @torch.no_grad()
    def _generate_labels(self):
        self.log_scale = torch.rand(self.num_tasks) * 2 - 1 + np.log(self.tanh_scale)  # uniform over [-1, 1]
        self.task_labels = torch.tanh((self.latents @ self.task_weights) * torch.exp(self.log_scale) + self.task_bias)  # (N, T)


class LatentSinDataset(LatentSparseLinearDataset):
    def __init__(self, latent_dim, num_samples, num_tasks, embedding_fn, bias=False, correlation=0.0, sparsity=0.0, task_correlation=0.0, sin_scale=1.0):
        assert sin_scale > 0, "sin_scale must be positive"
        self.sin_scale = sin_scale
        super().__init__(
            latent_dim=latent_dim, num_samples=num_samples, num_tasks=num_tasks, 
            embedding_fn=embedding_fn, bias=bias, correlation=correlation, sparsity=sparsity, task_correlation=task_correlation
        )

    @torch.no_grad()
    def _generate_labels(self):
        self.log_scale = torch.rand(self.num_tasks) * 2 - 1  # uniform over [-1, 1]
        self.task_labels = torch.sin((self.latents @ self.task_weights) * self.sin_scale * np.pi * torch.exp(self.log_scale) + self.task_bias)  # (N, T)


class LatentWaveletDataset(LatentSparseLinearDataset):
    def __init__(self, latent_dim, num_samples, num_tasks, embedding_fn, bias=False, correlation=0.0, sparsity=0.0, task_correlation=0.0, sin_scale=1.0, length_scale=1.0):
        assert length_scale > 0, "length_scale must be positive"
        self.length_scale = length_scale
        assert sin_scale > 0, "sin_scale must be positive"
        self.sin_scale = sin_scale
        super().__init__(
            latent_dim=latent_dim, num_samples=num_samples, num_tasks=num_tasks, 
            embedding_fn=embedding_fn, bias=bias, correlation=correlation, sparsity=sparsity, task_correlation=task_correlation
        )

    @torch.no_grad()
    def _generate_labels(self):
        self.log_scale = torch.rand(self.num_tasks) * 2 - 1  # uniform over [-1, 1]
        self.task_labels = (
            torch.sin((self.latents @ self.task_weights) * self.sin_scale * np.pi * torch.exp(self.log_scale) + self.task_bias) *
            torch.exp(-torch.square(self.latents @ self.task_weights) / (2 * self.length_scale ** 2))
        )  # (N, T)


class LatentInvDataset(LatentTanhDataset):
    def __init__(self, latent_dim, num_samples, num_tasks, embedding_fn, bias=False, correlation=0.0, sparsity=0.0, task_correlation=0.0):
        super().__init__(
            latent_dim=latent_dim, num_samples=num_samples, num_tasks=num_tasks, 
            embedding_fn=embedding_fn, bias=bias, correlation=correlation, sparsity=sparsity, task_correlation=task_correlation
        )

    @torch.no_grad()
    def _generate_labels(self):
        self.log_scale = torch.rand(self.num_tasks) * 2 - 1  # uniform over [-1, 1]
        self.task_labels = 1 / (torch.abs((self.latents @ self.task_weights) * torch.exp(self.log_scale) + self.task_bias) + 1)  # (N, T)


class LatentMLPDataset(torch.utils.data.Dataset):
    def __init__(self, latent_dim, num_samples, num_tasks, embedding_fn, sparsity=0.0, mlp_nonlinearity="relu", mlp_hidden_dim=64, correlation=0.0):
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.num_tasks = num_tasks
        self.embedding_fn = embedding_fn
        self.mlp_nonlinearity = mlp_nonlinearity
        self.mlp_hidden_dim = mlp_hidden_dim
        self.correlation = correlation

        if (1 - sparsity) < (1 / latent_dim):
            warnings.warn(
                f"Sparsity level {sparsity} is too high for latent dimension {latent_dim}, "
                f"must have at least 1 nonzero weight per task. Setting sparsity to {1 - 1/latent_dim}."
            )
            sparsity = 1 - 1/latent_dim
        self.sparsity = sparsity

        if correlation != 0.0:
            covariance = torch.eye(latent_dim) * (1 - correlation) + torch.ones((latent_dim, latent_dim)) * correlation
            self.data_dist = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(latent_dim), covariance
            )
        else:
            self.data_dist = torch.distributions.normal.Normal(
            torch.zeros(latent_dim), torch.ones(latent_dim)
        )

        self._init_data()

    @torch.no_grad()
    def _init_data(self):
        self.latents = self.data_dist.sample((self.num_samples,))  # (N, L)
        self.representations = self.embedding_fn(self.latents)  # (N, R)

        self.jacobian_mask = torch.zeros(self.latent_dim, self.num_tasks)  # (L, T)
        for t in range(self.num_tasks):
            if (1 - self.sparsity) <= (1 / self.latent_dim):
                num_nonzero = 1  # Ensure at least 1 nonzero weight
            else:
                num_nonzero = 1 + torch.binomial(
                    torch.tensor(self.latent_dim - 1.0),
                    torch.tensor((self.latent_dim * (1 - self.sparsity) - 1) / (self.latent_dim - 1)),
                ).int().item()
            nonzero_indices = torch.randperm(self.latent_dim)[:num_nonzero]
            self.jacobian_mask[nonzero_indices, t] = 1.0

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
            mlp.layers[0].weight.data *= self.jacobian_mask[:, t].unsqueeze(1)  # Mask the input weights to enforce sparsity
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
    embedding_type: str = "standard",  # "standard" or "sphere"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if embedding_type == "standard":
        model_cls = EmbeddingAutoencoder
    elif embedding_type == "sphere":
        model_cls = SphereEmbeddingAutoencoder
    else:
        raise ValueError(f"Invalid embedding_type: {embedding_type}")

    model = model_cls(
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


class SphereEmbeddingAutoencoder(nn.Module):
    def __init__(self, input_dim, representation_dim, encoder_hidden_dims=[], decoder_hidden_dims=[], noise_std=0.01, activation="relu"):
        super(SphereEmbeddingAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims
        self.noise_std = noise_std
        self.activation = activation

        self.base_encoder = EmbeddingEncoder(input_dim, representation_dim - 1, encoder_hidden_dims, noise_std, activation)
        self.decoder = EmbeddingDecoder(representation_dim, input_dim, decoder_hidden_dims, activation)
    
    def encoder(self, x):
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        x_normalized = x / x_norm  # Project input onto sphere
        z = self.base_encoder(x_normalized)
        z = torch.cat([z, x_norm], dim=-1)  # Append norm as additional dimension
        return z
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z