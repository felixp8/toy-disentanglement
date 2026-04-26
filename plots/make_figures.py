#!/usr/bin/env python3
"""Generate report figures from the wandb project felixp8/toy-disentanglement.

Produces four PDFs in report/figures/:
- unbiased.pdf   : r2 and acc_ub vs P, per depth, for sweep_id = 1
- biased.pdf     : same for sweep_id = 2
- tanh.pdf       : same for sweep_id = 5
- compare.pdf    : three task families overlaid at depth k = 4
"""
from pathlib import Path

import matplotlib.pyplot as plt
import wandb

ROOT = Path(__file__).parent
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

ENTITY = "felixp8"
PROJECT = "toy-disentanglement"
P_VALUES = [2, 4, 6, 8, 10, 15, 20, 30, 50]
DEPTHS = [1, 2, 3, 4]


def pull_sweep(sweep_id):
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"config.sweep_id": sweep_id})
    out = {}
    for r in runs:
        if r.state != "finished":
            continue
        P = r.config.get("dataset", {}).get("num_tasks")
        k = len(r.config.get("model", {}).get("hidden_dims", []))
        # r2 = r.summary.get("r2/test_last")
        metrics = r.history()
        if P is None or k is None or metrics['r2/test_last'].iloc[-1] is None:
            continue
        out[(P, k)] = {
            "r2":       float(metrics["r2/test_last"].iloc[-10:].mean()),
            "r2_last":  float(metrics["r2/test_last"].iloc[-10:].mean()),
            "r2_max":   _maybe_float(metrics["r2/test_max"].iloc[-10:].mean() if "r2/test_max" in metrics else None),
            "r2_argmax": _maybe_int(metrics["r2/test_argmax"].iloc[-10:].mode()[0] if "r2/test_argmax" in metrics else None),
            "acc_ub":   _maybe_float(metrics["acc_ub/test_last"].iloc[-10:].mean() if "acc_ub/test_last" in metrics else None),
            "idim_last": _maybe_float(metrics["intrinsic_dim/last"].iloc[-10:].mean() if "intrinsic_dim/last" in metrics else None),
            "idim_max":  _maybe_float(metrics["intrinsic_dim/max"].iloc[-10:].mean() if "intrinsic_dim/max" in metrics else None),
            "idim_min":  _maybe_float(metrics["intrinsic_dim/min"].iloc[-10:].mean() if "intrinsic_dim/min" in metrics else None),
        }
    return out


def _maybe_float(v):
    return float(v) if v is not None else None


def _maybe_int(v):
    return int(v) if v is not None else None


def depth_colors():
    cmap = plt.get_cmap("viridis")
    return {d: cmap(i / (len(DEPTHS) - 1)) for i, d in enumerate(DEPTHS)}


def _style_axes(axes):
    for ax in axes:
        ax.axvline(5, color="0.6", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("# training tasks $P$")
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 50])
        ax.set_xticklabels(["2", "5", "10", "20", "50"])
        ax.tick_params(axis="both", labelsize=9)
    axes[0].set_ylabel("regression generalization\n" + r"($R^2$)")
    axes[0].set_ylim(-0.55, 1.05)
    axes[0].axhline(0, color="0.7", linewidth=0.6)
    axes[1].set_ylabel("classifier generalization\n(accuracy)")
    axes[1].set_ylim(0.48, 1.02)
    axes[1].axhline(0.5, color="0.7", linewidth=0.6)


def plot_family(data, title, path):
    colors = depth_colors()
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.1))
    for k in DEPTHS:
        xs, r2s, accs = [], [], []
        for P in P_VALUES:
            if (P, k) in data:
                xs.append(P)
                r2s.append(data[(P, k)]["r2"])
                accs.append(data[(P, k)]["acc_ub"])
        axes[0].plot(xs, r2s, marker="o", ms=4, color=colors[k], label=f"$k = {k}$")
        axes[1].plot(xs, accs, marker="o", ms=4, color=colors[k], label=f"$k = {k}$")
    _style_axes(axes)
    axes[0].legend(loc="lower right", fontsize=8, frameon=False, title="depth", title_fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_compare(datas, path):
    families = [
        ("zero-threshold classification", datas[1], "#1f77b4"),
        ("shifted-threshold classification",   datas[2], "#d62728"),
        ("tanh",           datas[5], "#2ca02c"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.1))
    for name, data, color in families:
        xs, r2s, accs = [], [], []
        for P in P_VALUES:
            if (P, 4) in data:
                xs.append(P)
                r2s.append(data[(P, 4)]["r2"])
                accs.append(data[(P, 4)]["acc_ub"])
        axes[0].plot(xs, r2s, marker="o", ms=4, color=color, label=name)
        axes[1].plot(xs, accs, marker="o", ms=4, color=color, label=name)
    _style_axes(axes)
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(r"Fixed depth $k = 4$", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_fig3e_reproduction(data, path, depth=2):
    """Reproduction of paper Fig 3e: classifier and regression generalization vs P, at fixed depth.

    Uses the unbiased-classification sweep results at a shallow depth so the
    saturation regime reported in paper Fig 3e is recovered cleanly.
    """
    xs, r2s, accs = [], [], []
    for P in P_VALUES:
        if (P, depth) in data:
            xs.append(P)
            r2s.append(data[(P, depth)]["r2_last"])
            accs.append(data[(P, depth)]["acc_ub"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    axes[0].plot(xs, accs, marker="o", color="#1f77b4", ms=5)
    axes[0].axvline(5, color="0.6", linestyle="--", lw=0.8, alpha=0.7)
    axes[0].axhline(0.5, color="0.7", lw=0.5)
    axes[0].set_xscale("log")
    axes[0].set_xticks([2, 5, 10, 20, 50]); axes[0].set_xticklabels(["2", "5", "10", "20", "50"])
    axes[0].set_xlabel("# tasks $P$")
    axes[0].set_ylabel("classifier generalization\n(accuracy)")
    axes[0].set_ylim(0.48, 1.02)
    axes[0].set_title("Classifier generalization")

    axes[1].plot(xs, r2s, marker="o", color="#1f77b4", ms=5)
    axes[1].axvline(5, color="0.6", linestyle="--", lw=0.8, alpha=0.7)
    axes[1].axhline(0, color="0.7", lw=0.5)
    axes[1].set_xscale("log")
    axes[1].set_xticks([2, 5, 10, 20, 50]); axes[1].set_xticklabels(["2", "5", "10", "20", "50"])
    axes[1].set_xlabel("# tasks $P$")
    axes[1].set_ylabel("regression generalization\n" + r"($R^2$)")
    axes[1].set_ylim(-0.1, 1.02)
    axes[1].set_title("Regression generalization")

    fig.suptitle(rf"Saturation of abstraction metrics with $P$ (depth $k = {depth}$)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_per_layer(datas, path, P_target=50):
    """Three-panel: r2_last vs r2_max as a function of depth, per family, at fixed P."""
    families = [
        ("Zero-threshold classification", datas[1], "#1f77b4"),
        ("Shifted-threshold classification",   datas[2], "#d62728"),
        ("Tanh",           datas[5], "#2ca02c"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.1), sharey=True)
    for ax, (name, data, color) in zip(axes, families):
        xs, last, mx = [], [], []
        for k in DEPTHS:
            if (P_target, k) in data:
                xs.append(k)
                last.append(data[(P_target, k)]["r2_last"])
                mx.append(data[(P_target, k)]["r2_max"])
        ax.plot(xs, mx,   marker="^", ms=6, color=color, linestyle="--", label=r"best layer ($r^2_{\mathrm{max}}$)")
        ax.plot(xs, last, marker="o", ms=6, color=color, linestyle="-",  label=r"last layer ($r^2_{\mathrm{last}}$)")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("depth $k$")
        ax.set_xticks(DEPTHS)
        ax.set_ylim(-0.15, 1.05)
        ax.axhline(0, color="0.7", linewidth=0.5)
        ax.tick_params(axis="both", labelsize=9)
    axes[0].set_ylabel("regression generalization\n" + r"($R^2$)")
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle(rf"Best-layer vs. last-layer regression generalization at $P = {P_target}$", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Pulling sweeps from wandb...")
    datas = {sid: pull_sweep(sid) for sid in (1, 2, 5)}
    for sid, d in datas.items():
        print(f"  sweep {sid}: {len(d)} cells")

    plot_family(datas[1], "Zero-threshold classification",                FIGS / "unbiased.pdf")
    plot_family(datas[2], "Shifted-threshold classification",                  FIGS / "biased.pdf")
    plot_family(datas[5], "Tanh (continuous target)",      FIGS / "tanh.pdf")
    plot_compare(datas, FIGS / "compare.pdf")
    plot_per_layer(datas, FIGS / "per_layer.pdf")
    plot_fig3e_reproduction(datas[1], FIGS / "fig3_abstraction_vs_P.pdf", depth=2)

    for p in sorted(FIGS.glob("*.pdf")):
        print("wrote", p)


if __name__ == "__main__":
    main()
