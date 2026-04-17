import numpy as np
import torch

from sklearn.linear_model import LogisticRegression, Ridge


@torch.no_grad()
def classification_generalization_accuracy(
    rep_fn,
    data_dist,
    embed_fn,
    task_type: str = "random", # {"random", "axis-aligned"}
    num_tasks: int = 100,
    split_orthogonal: bool = True,
    bias: bool = False,
):
    latent_dim = data_dist.sample((1,)).shape[-1]

    train_accuracies = []
    test_accuracies = []
    for _ in range(num_tasks):
        samples = data_dist.sample((1000,))
        embeddings = embed_fn(samples)
        representations = rep_fn(embeddings)

        if task_type == "random":
            task_weight = torch.randn(latent_dim)
        elif task_type == "axis-aligned":
            task_weight = torch.zeros(latent_dim)
            task_weight[torch.randint(latent_dim)] = 1
        if bias:
            task_bias = torch.randn(1) * 0.3
        else:
            task_bias = torch.zeros(1)
        task_weight = task_weight / torch.norm(task_weight)
        labels = torch.sign(samples @ task_weight + task_bias)

        split_weight = torch.randn(latent_dim)
        split_weight = split_weight / torch.norm(split_weight)
        if split_orthogonal:
            # project split axis onto task axis
            split_proj = split_weight @ task_weight
            split_weight -= split_proj * task_weight
            split_weight = split_weight / torch.norm(split_weight)
        train_mask = (samples @ split_weight) >= 0
        test_mask = (samples @ split_weight) < 0

        if isinstance(representations, list):
            layer_train_accuracies = []
            layer_test_accuracies = []
            for i in range(len(representations)):
                train_representations = representations[i][train_mask]
                test_representations = representations[i][test_mask]
                clf = LogisticRegression(max_iter=1000)
                clf.fit(train_representations, labels[train_mask])

                train_acc = clf.score(train_representations, labels[train_mask])
                test_acc = clf.score(test_representations, labels[test_mask])

                layer_train_accuracies.append(train_acc)
                layer_test_accuracies.append(test_acc)
            train_accuracies.append(np.array(layer_train_accuracies))
            test_accuracies.append(np.array(layer_test_accuracies))
        else:
            train_representations = representations[train_mask]
            train_labels = labels[train_mask]
            test_representations = representations[test_mask]
            test_labels = labels[test_mask]

            clf = LogisticRegression(max_iter=1000)
            clf.fit(train_representations, train_labels)

            train_acc = clf.score(train_representations, train_labels)
            test_acc = clf.score(test_representations, test_labels)

            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
    train_accuracies = np.array(train_accuracies)
    test_accuracies = np.array(test_accuracies)

    if train_accuracies.ndim == 1:
        train_accuracies = train_accuracies[:, np.newaxis]
        test_accuracies = test_accuracies[:, np.newaxis]

    return train_accuracies, test_accuracies

@torch.no_grad()
def regression_generalization_r2(
    rep_fn,
    data_dist,
    embed_fn,
    task_type: str = "random", # {"random", "axis-aligned"}
    num_tasks: int = 100,
    split_orthogonal: bool = True,
):
    latent_dim = data_dist.sample((1,)).shape[-1]

    train_r2s = []
    test_r2s = []
    for _ in range(num_tasks):
        samples = data_dist.sample((1000,))
        embeddings = embed_fn(samples)
        representations = rep_fn(embeddings)

        if task_type == "random":
            task_weight = torch.randn(latent_dim)
        elif task_type == "axis-aligned":
            task_weight = torch.zeros(latent_dim)
            task_weight[torch.randint(latent_dim)] = 1
        task_weight = task_weight / torch.norm(task_weight)
        targets = samples @ task_weight

        split_weight = torch.randn(latent_dim)
        split_weight = split_weight / torch.norm(split_weight)
        if split_orthogonal:
            # project split axis onto task axis
            split_proj = split_weight @ task_weight
            split_weight -= split_proj * task_weight
            split_weight = split_weight / torch.norm(split_weight)
        train_mask = (samples @ split_weight) >= 0
        test_mask = (samples @ split_weight) < 0

        if isinstance(representations, list):
            layer_train_r2s = []
            layer_test_r2s = []
            for i in range(len(representations)):
                train_representations = representations[i][train_mask]
                test_representations = representations[i][test_mask]

                reg = Ridge(max_iter=1000)
                reg.fit(train_representations, targets[train_mask])

                train_r2 = reg.score(train_representations, targets[train_mask])
                test_r2 = reg.score(test_representations, targets[test_mask])

                layer_train_r2s.append(train_r2)
                layer_test_r2s.append(test_r2)
            train_r2s.append(np.array(layer_train_r2s))
            test_r2s.append(np.array(layer_test_r2s))
        else:
            train_representations = representations[train_mask]
            train_targets = targets[train_mask]
            test_representations = representations[test_mask]
            test_targets = targets[test_mask]

            reg = Ridge(max_iter=1000)
            reg.fit(train_representations, train_targets)

            train_r2 = reg.score(train_representations, train_targets)
            test_r2 = reg.score(test_representations, test_targets)

            train_r2s.append(train_r2)
            test_r2s.append(test_r2)
    train_r2s = np.array(train_r2s)
    test_r2s = np.array(test_r2s)

    if train_r2s.ndim == 1:
        train_r2s = train_r2s[:, np.newaxis]
        test_r2s = test_r2s[:, np.newaxis]

    return train_r2s, test_r2s