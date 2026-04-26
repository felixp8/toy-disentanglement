#!/usr/bin/env python3
"""Reproduce Johnston & Fusi (2023) Fig. 2 and Fig. 3 using our PyTorch pipeline.

Trains:
- one input autoencoder with the same $L_2$ + participation-ratio regularization
- three multi-tasking MLPs on unbiased classification at P = 1, 2, 10

Writes the following PDFs into report/figures/:
- fig2_response_fields.pdf  (paper Fig 2b)
- fig2_concentric.pdf        (paper Fig 2c)
- fig2_sparseness_pr.pdf     (paper Fig 2d)
- fig2_metric_viz.pdf        (paper Fig 2e)
- fig3_concentric.pdf        (paper Fig 3b)
- fig3_metric_viz.pdf        (paper Fig 3c)
- fig3_task_projection.pdf   (paper Fig 3d)
"""
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
from toy_disentanglement.task import (  # noqa: E402
    create_embedding_autoencoder,
    LatentClassificationDataset,
)
from toy_disentanglement.model import MLP  # noqa: E402

FIGS = REPO_ROOT / "report" / "figures"
FIGS.mkdir(exist_ok=True)

D = 5
REP_DIM = 128
MLP_WIDTH = 256
MLP_DEPTH = 2
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------------
# training
# ------------------------------------------------------------------

def train_autoencoder():
    print("Training autoencoder...")
    ae = create_embedding_autoencoder(
        input_dim=D,
        representation_dim=REP_DIM,
        encoder_hidden_dims=[128, 128],
        decoder_hidden_dims=[128, 128],
        noise_std=0.0,
        activation="relu",
        train=True,
        num_epochs=2000,
        batch_size=256,
        learning_rate=1e-3,
        l2_penalty_weight=0.1,
        pr_penalty_weight=0.1,
        verbose=False,
        checkpoint_path=None,
    )
    ae.eval()
    return ae


def train_multi_tasking(encoder, P, num_epochs=100):
    print(f"Training multi-tasking MLP for P = {P} ...")
    ds = LatentClassificationDataset(
        latent_dim=D, num_samples=16384, num_tasks=P,
        embedding_fn=encoder, bias=False, sparsity=0.0,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    mlp = MLP(
        input_dim=REP_DIM, output_dim=P,
        hidden_dims=[MLP_WIDTH] * MLP_DEPTH,
        activation="relu", output_activation="tanh",
    )
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-4, weight_decay=1e-6)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        mlp.train()
        for batch in loader:
            latents, reps, labels = batch
            opt.zero_grad()
            pred = mlp(reps)
            loss = loss_fn(pred, labels.float())
            loss.backward()
            opt.step()
    mlp.eval()
    return mlp, ds


# ------------------------------------------------------------------
# stimulus / PCA utilities
# ------------------------------------------------------------------

def concentric_square(n_per_side=80, radii=(0.4, 0.8, 1.2, 1.6, 2.0)):
    """Five concentric squares in the latent[0,1] plane, latent[2:D] = 0."""
    parts, cs = [], []
    for i, r in enumerate(radii):
        t = np.linspace(-1, 1, n_per_side, endpoint=False)
        top    = np.column_stack([t * r,               np.full_like(t,  r)])
        right  = np.column_stack([np.full_like(t,  r), -t * r])
        bottom = np.column_stack([-t * r,              np.full_like(t, -r)])
        left   = np.column_stack([np.full_like(t, -r), t * r])
        square = np.concatenate([top, right, bottom, left], axis=0)
        pts = np.zeros((square.shape[0], D))
        pts[:, :2] = square
        parts.append(pts)
        cs.append(np.full(square.shape[0], i))
    return np.concatenate(parts, 0).astype(np.float32), np.concatenate(cs, 0)


def fit_pca(z, k=3):
    pca = PCA(n_components=k)
    pca.fit(z)
    return pca


@torch.no_grad()
def encode(encoder, x_np):
    return encoder(torch.as_tensor(x_np, dtype=torch.float32)).numpy()


@torch.no_grad()
def rep_layer(encoder, mlp, x_np):
    z = encoder(torch.as_tensor(x_np, dtype=torch.float32))
    return mlp.get_all_layer_representations(z)[-1].numpy()


# ------------------------------------------------------------------
# Fig 2 panels
# ------------------------------------------------------------------

def fig2_response_fields(encoder, path, grid_size=40, n_units=25, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.linspace(-3, 3, grid_size)
    mesh = np.stack(np.meshgrid(xs, xs, indexing="ij"), axis=-1)
    pts = np.zeros((grid_size * grid_size, D), dtype=np.float32)
    pts[:, :2] = mesh.reshape(-1, 2)
    z = encode(encoder, pts).reshape(grid_size, grid_size, REP_DIM)
    unit_idx = rng.choice(REP_DIM, size=n_units, replace=False)
    fig, axes = plt.subplots(5, 5, figsize=(5.3, 5.3))
    for ax, u in zip(axes.flat, unit_idx):
        ax.imshow(z[:, :, u].T, origin="lower", cmap="viridis",
                  extent=(-3, 3, -3, 3))
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(r"Response fields of 25 random encoder units (latent$_0$, latent$_1$)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig2_concentric(encoder, path):
    lat, colors = concentric_square()
    rng = np.random.default_rng(0)
    x_ref = rng.standard_normal((5000, D)).astype(np.float32)
    z_ref = encode(encoder, x_ref)
    z_sq = encode(encoder, lat)
    pca = fit_pca(z_ref)
    proj_lat = lat[:, :2]
    proj_enc = pca.transform(z_sq)

    fig = plt.figure(figsize=(7, 3.2))
    ax0 = fig.add_subplot(1, 2, 1)
    ax0.scatter(proj_lat[:, 0], proj_lat[:, 1], c=colors, cmap="viridis", s=2)
    ax0.set_title("Latent variables")
    ax0.set_aspect("equal")
    ax0.set_xlabel(r"latent$_0$"); ax0.set_ylabel(r"latent$_1$")

    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.scatter(proj_enc[:, 0], proj_enc[:, 1], proj_enc[:, 2],
                c=colors, cmap="viridis", s=2)
    ax1.set_title("After input encoder (PC1--3)")
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2"); ax1.set_zlabel("PC3")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig2_sparseness_pr(encoder, path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5000, D)).astype(np.float32)
    z = encode(encoder, x)

    mean_r = z.mean(0)
    sq_mean_r = (z ** 2).mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        S_per_unit = 1 - mean_r ** 2 / sq_mean_r
    S_enc = float(np.nanmean(S_per_unit))
    S_lat = 0.0

    C = np.cov(z, rowvar=False)
    eigs = np.linalg.eigvalsh(C)
    pr_enc = float((eigs.sum() ** 2) / (eigs ** 2).sum())
    pr_lat = float(D)

    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.7))
    axes[0].bar([0, 1], [S_lat, S_enc], color=["#2ca02c", "#34495e"], width=0.6)
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["latents", "input"])
    axes[0].set_ylabel("per-unit sparseness")
    axes[0].set_ylim(0, 1.02)
    axes[1].bar([0, 1], [pr_lat, pr_enc], color=["#2ca02c", "#34495e"], width=0.6)
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["latents", "input"])
    axes[1].set_ylabel("participation ratio")
    fig.suptitle(f"Input sparseness $S={S_enc:.2f}$, embedding PR $={pr_enc:.0f}$", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig2_metric_viz(encoder, path):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2000, D)).astype(np.float32)
    z = encode(encoder, x)
    w = rng.standard_normal(D); w /= np.linalg.norm(w)
    s = rng.standard_normal(D); s -= (s @ w) * w; s /= np.linalg.norm(s)
    proj_split = x @ s
    proj_task = x @ w
    train_mask = proj_split < 0   # train on LEFT half so it matches paper Fig 2e convention

    clf = LogisticRegression(max_iter=500)
    clf.fit(z[train_mask], np.sign(proj_task[train_mask]))
    clf_pred = clf.decision_function(z)

    reg = Ridge()
    reg.fit(z[train_mask], proj_task[train_mask])
    reg_pred = reg.predict(z)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))
    axes[0].scatter(proj_split, clf_pred, c=np.sign(proj_task),
                    cmap="coolwarm", s=4, alpha=0.7)
    axes[0].axvline(0, color="k", linestyle="--", lw=0.8)
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].set_xlabel(r"split direction $x \cdot s$")
    axes[0].set_ylabel("classifier decision")
    axes[0].set_title("Classifier generalization")
    ylim = axes[0].get_ylim()
    axes[0].text(-2.8, ylim[1] * 0.85, "trained", fontsize=9, color="0.3")
    axes[0].text( 1.4, ylim[1] * 0.85, "tested",  fontsize=9, color="0.3")

    axes[1].scatter(proj_split, reg_pred, c=proj_task,
                    cmap="viridis", s=4, alpha=0.7)
    axes[1].axvline(0, color="k", linestyle="--", lw=0.8)
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_xlabel(r"split direction $x \cdot s$")
    axes[1].set_ylabel("regression estimate")
    axes[1].set_title("Regression generalization")
    fig.suptitle("Encoder-layer abstraction metrics", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# Fig 3 panels
# ------------------------------------------------------------------

def fig3_concentric(encoder, mlps, path):
    lat, colors = concentric_square()
    rng = np.random.default_rng(0)
    x_ref = rng.standard_normal((5000, D)).astype(np.float32)

    fig = plt.figure(figsize=(10, 3.4))
    for i, (P, mlp) in enumerate(mlps.items()):
        rep_ref = rep_layer(encoder, mlp, x_ref)
        rep_sq = rep_layer(encoder, mlp, lat)
        pca = fit_pca(rep_ref)
        proj = pca.transform(rep_sq)
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
                   c=colors, cmap="viridis", s=2)
        ax.set_title(f"$P = {P}$")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    fig.suptitle("Representation-layer geometry as $P$ increases", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig3_task_projection(encoder, mlps, datasets, path):
    fig, axes = plt.subplots(1, len(mlps), figsize=(9, 2.8), sharey=True)
    rng = np.random.default_rng(2)
    for ax, (P, mlp) in zip(axes, mlps.items()):
        ds = datasets[P]
        w_task = ds.task_weights[:, 0].detach().numpy()
        x = rng.standard_normal((2000, D)).astype(np.float32)
        labels = np.sign(x @ w_task)
        rep = rep_layer(encoder, mlp, x)
        readout = mlp.layers[-2]  # last Linear (tanh is layers[-1])
        w_out = readout.weight.detach().numpy()[0]
        b_out = readout.bias.detach().numpy()[0]
        proj = rep @ w_out + b_out
        ax.hist(proj[labels > 0], bins=25, alpha=0.6, color="#d62728", label="$+1$")
        ax.hist(proj[labels < 0], bins=25, alpha=0.6, color="#1f77b4", label="$-1$")
        ax.set_title(f"$P = {P}$")
        ax.set_xlabel("projection onto task-0 readout")
    axes[0].set_ylabel("count")
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("Training-task readout histogram (task 0)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig3_sample_efficiency(
    encoder, mlp, path,
    n_samples_list=(5, 10, 20, 50, 100, 200, 500, 1000),
    n_repeats=12, n_test=1000,
):
    """Reproduce paper Fig 3f: novel-task sample efficiency and generalization.

    For each number of training samples $n$, we draw a novel random classification task and
    fit a logistic regression on three feature sets:
    - the latent variables directly (upper bound);
    - the multi-tasking MLP's representation layer;
    - the input encoder's representation (lower bound).

    Two panels: left, standard performance (train and test on the whole latent space);
    right, generalization performance (train on one half, test on the other).
    """
    rng = np.random.default_rng(10)
    conds = ("latents", "mtl", "input")
    results = {split: {c: [] for c in conds} for split in ("standard", "gen")}

    for n in n_samples_list:
        acc = {c: {"standard": [], "gen": []} for c in conds}
        for _ in range(n_repeats):
            w = rng.standard_normal(D); w /= np.linalg.norm(w)
            s = rng.standard_normal(D); s -= (s @ w) * w; s /= np.linalg.norm(s)

            # ---- standard: train on n random samples, test on n_test fresh samples ----
            x_train = rng.standard_normal((n, D)).astype(np.float32)
            x_test  = rng.standard_normal((n_test, D)).astype(np.float32)
            y_train = np.sign(x_train @ w)
            y_test  = np.sign(x_test @ w)
            if len(np.unique(y_train)) >= 2:
                for c in conds:
                    f_train = _feat(c, encoder, mlp, x_train)
                    f_test  = _feat(c, encoder, mlp, x_test)
                    clf = LogisticRegression(max_iter=500)
                    clf.fit(f_train, y_train)
                    acc[c]["standard"].append(clf.score(f_test, y_test))

            # ---- generalization: train on half-space, test on the other half ----
            pool_train = rng.standard_normal((max(n * 10, 200), D)).astype(np.float32)
            x_train_g = pool_train[(pool_train @ s) < 0][:n]
            pool_test  = rng.standard_normal((n_test * 5, D)).astype(np.float32)
            x_test_g   = pool_test[(pool_test @ s) >= 0][:n_test]
            if x_train_g.shape[0] < n or x_test_g.shape[0] == 0:
                continue
            y_train_g = np.sign(x_train_g @ w)
            y_test_g  = np.sign(x_test_g @ w)
            if len(np.unique(y_train_g)) < 2:
                continue
            for c in conds:
                f_train = _feat(c, encoder, mlp, x_train_g)
                f_test  = _feat(c, encoder, mlp, x_test_g)
                clf = LogisticRegression(max_iter=500)
                clf.fit(f_train, y_train_g)
                acc[c]["gen"].append(clf.score(f_test, y_test_g))

        for c in conds:
            results["standard"][c].append(np.mean(acc[c]["standard"]) if acc[c]["standard"] else np.nan)
            results["gen"][c].append(     np.mean(acc[c]["gen"])      if acc[c]["gen"]      else np.nan)

    # ---- plot ----
    colors = {"latents": "#999999", "mtl": "#1f77b4", "input": "#333333"}
    styles = {"latents": ":",       "mtl": "-",       "input": "--"}
    labels = {"latents": "latents (upper bound)", "mtl": "multi-tasking rep", "input": "input (lower bound)"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for c in conds:
        axes[0].plot(n_samples_list, results["standard"][c],
                     color=colors[c], linestyle=styles[c], marker="o", ms=4, label=labels[c])
        axes[1].plot(n_samples_list, results["gen"][c],
                     color=colors[c], linestyle=styles[c], marker="o", ms=4, label=labels[c])
    for ax, title in zip(axes, ("Standard", "Generalization")):
        ax.set_xscale("log")
        ax.set_xlabel("number of training samples $n$")
        ax.axhline(0.5, color="0.7", lw=0.5)
        ax.set_title(title)
        ax.set_ylim(0.48, 1.02)
    axes[0].set_ylabel("novel-task accuracy")
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(r"Sample-efficient learning of a novel task ($P = 10$ multi-tasking model)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _feat(name, encoder, mlp, x_np):
    if name == "latents":
        return x_np
    elif name == "mtl":
        return rep_layer(encoder, mlp, x_np)
    elif name == "input":
        return encode(encoder, x_np)
    else:
        raise ValueError(name)


def fig3_metric_viz(encoder, mlps, path):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2000, D)).astype(np.float32)
    w = rng.standard_normal(D); w /= np.linalg.norm(w)
    s = rng.standard_normal(D); s -= (s @ w) * w; s /= np.linalg.norm(s)
    proj_split = x @ s
    proj_task = x @ w
    train_mask = proj_split < 0   # train on LEFT half so it matches paper Fig 2e convention

    fig, axes = plt.subplots(2, len(mlps), figsize=(9, 5.6), sharex=True)
    for col, (P, mlp) in enumerate(mlps.items()):
        rep = rep_layer(encoder, mlp, x)
        clf = LogisticRegression(max_iter=500)
        clf.fit(rep[train_mask], np.sign(proj_task[train_mask]))
        clf_pred = clf.decision_function(rep)
        axes[0, col].scatter(proj_split, clf_pred, c=np.sign(proj_task),
                             cmap="coolwarm", s=4, alpha=0.7)
        axes[0, col].axvline(0, color="k", linestyle="--", lw=0.8)
        axes[0, col].axhline(0, color="gray", lw=0.5)
        axes[0, col].set_title(f"$P = {P}$")
        if col == 0:
            axes[0, col].set_ylabel("classifier decision")

        reg = Ridge()
        reg.fit(rep[train_mask], proj_task[train_mask])
        reg_pred = reg.predict(rep)
        axes[1, col].scatter(proj_split, reg_pred, c=proj_task,
                             cmap="viridis", s=4, alpha=0.7)
        axes[1, col].axvline(0, color="k", linestyle="--", lw=0.8)
        axes[1, col].axhline(0, color="gray", lw=0.5)
        axes[1, col].set_xlabel(r"split direction $x \cdot s$")
        if col == 0:
            axes[1, col].set_ylabel("regression estimate")
    fig.suptitle("Abstraction metrics on the multi-tasking representation layer", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main():
    ae = train_autoencoder()
    print("\n--- generating Fig 2 panels ---")
    fig2_response_fields(ae.encoder, FIGS / "fig2_response_fields.pdf")
    fig2_concentric(     ae.encoder, FIGS / "fig2_concentric.pdf")
    fig2_sparseness_pr(  ae.encoder, FIGS / "fig2_sparseness_pr.pdf")
    fig2_metric_viz(     ae.encoder, FIGS / "fig2_metric_viz.pdf")

    mlps, datasets = {}, {}
    for P in (1, 2, 10):
        mlp, ds = train_multi_tasking(ae.encoder, P)
        mlps[P] = mlp
        datasets[P] = ds

    print("\n--- generating Fig 3 panels ---")
    fig3_concentric(      ae.encoder, mlps,         FIGS / "fig3_concentric.pdf")
    fig3_task_projection( ae.encoder, mlps, datasets, FIGS / "fig3_task_projection.pdf")
    fig3_metric_viz(      ae.encoder, mlps,         FIGS / "fig3_metric_viz.pdf")
    fig3_sample_efficiency(ae.encoder, mlps[10],    FIGS / "fig3_sample_efficiency.pdf")

    for p in sorted(FIGS.glob("fig[23]_*.pdf")):
        print("wrote", p)


if __name__ == "__main__":
    main()
