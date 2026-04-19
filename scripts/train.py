#!/usr/bin/env python3

import numpy as np
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
import time

from toy_disentanglement.metrics import classification_generalization_accuracy, regression_generalization_r2, representation_intrinsic_dimension

ROOT = Path(__file__).parent.parent


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    latent_dim = cfg.embedding.input_dim
    batch_size = cfg.batch_size
    num_epochs = cfg.num_epochs
    lr = cfg.lr
    weight_decay = cfg.weight_decay
    eval_freq = cfg.eval_freq
    chkpt_freq = cfg.chkpt_freq
    verbose = cfg.verbose
    patience = cfg.patience
    
    run_dir = ROOT / "runs" / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=run_dir / "config.yaml")
    if chkpt_freq > 0:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    run = None
    if cfg.wandb:
        run = wandb.init(project="toy-disentanglement", config=OmegaConf.to_container(cfg), name=cfg.run_name, dir=str(run_dir), reinit="create_new")

    embedding_network = instantiate(cfg.embedding)
    torch.save(embedding_network.state_dict(), run_dir / "embedding_network.pt")
    dataset = instantiate(cfg.dataset, embedding_fn=embedding_network.encoder)

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))]
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # model = MLP(representation_dim, num_tasks, [representation_dim]*3, activation="relu")
    model = instantiate(cfg.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    data_dist = torch.distributions.normal.Normal(torch.zeros(latent_dim), torch.ones(latent_dim))

    if run is not None:
        num_params = sum(p.numel() for p in model.parameters())
        run.config.update({"num_params": num_params})

    train_loss_history = []
    val_loss_history = []
    metric_history = []
    counter = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            latents, representations, labels = batch
            optimizer.zero_grad()
            preds = model(representations)
            loss = criterion(preds, labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        train_loss_history.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_dataloader:
                latents, representations, labels = batch
                preds = model(representations)
                loss = criterion(preds, labels.float())
                val_loss += loss.item()
            val_loss /= len(val_dataloader)
            val_loss_history.append(val_loss)

            if val_loss <= min(val_loss_history):
                counter = 0
            else:
                counter += 1
            
            if patience > 0:
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch} with best val loss {min(val_loss_history):.4f}", flush=True)
                    break
        
        # if run is not None:
        #     run.log({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        if eval_freq > 0 and epoch % eval_freq == 0:
            if run is not None:
                run.log({"loss/train": train_loss, "loss/val": val_loss}, step=epoch)

            train_acc_ub, test_acc_ub = classification_generalization_accuracy(
                rep_fn=model.get_all_layer_representations,
                embed_fn=embedding_network.encoder,
                data_dist=data_dist,
                num_tasks=10,
                bias=False,
            )
            train_acc_b, test_acc_b = classification_generalization_accuracy(
                rep_fn=model.get_all_layer_representations,
                embed_fn=embedding_network.encoder,
                data_dist=data_dist,
                num_tasks=10,
                bias=True,
            )
            train_r2, test_r2 = regression_generalization_r2(
                rep_fn=model.get_all_layer_representations,
                embed_fn=embedding_network.encoder,
                data_dist=data_dist,
                num_tasks=10,
            )
            intrinsic_dimensions = representation_intrinsic_dimension(
                rep_fn=model.get_all_layer_representations,
                embed_fn=embedding_network.encoder,
                data_dist=data_dist,
            )
            train_acc_ub = np.nanmean(train_acc_ub, axis=0)
            test_acc_ub = np.nanmean(test_acc_ub, axis=0)
            train_acc_b = np.nanmean(train_acc_b, axis=0)
            test_acc_b = np.nanmean(test_acc_b, axis=0)
            train_r2 = np.nanmean(train_r2, axis=0)
            test_r2 = np.nanmean(test_r2, axis=0)
            metric_history.append({
                "train_acc_ub": train_acc_ub,
                "test_acc_ub": test_acc_ub,
                "train_acc_b": train_acc_b,
                "test_acc_b": test_acc_b,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "intrinsic_dimensions": intrinsic_dimensions,
            })

            if run is not None:
                run.log({
                    "acc_ub/train_max": train_acc_ub.max(),
                    "acc_ub/test_max": test_acc_ub.max(),
                    "acc_ub/train_last": train_acc_ub[-1],
                    "acc_ub/test_last": test_acc_ub[-1],
                    "acc_ub/train_median": np.nanmedian(train_acc_ub),
                    "acc_ub/test_median": np.nanmedian(test_acc_ub),
                    "acc_ub/train_argmax": np.argmax(train_acc_ub),
                    "acc_ub/test_argmax": np.argmax(test_acc_ub),
                    "acc_b/train_max": train_acc_b.max(),
                    "acc_b/test_max": test_acc_b.max(),
                    "acc_b/train_last": train_acc_b[-1],
                    "acc_b/test_last": test_acc_b[-1],
                    "acc_b/train_median": np.nanmedian(train_acc_b),
                    "acc_b/test_median": np.nanmedian(test_acc_b),
                    "acc_b/train_argmax": np.argmax(train_acc_b),
                    "acc_b/test_argmax": np.argmax(test_acc_b),
                    "r2/train_max": train_r2.max(),
                    "r2/test_max": test_r2.max(),
                    "r2/train_last": train_r2[-1],
                    "r2/test_last": test_r2[-1],
                    "r2/train_median": np.nanmedian(train_r2),
                    "r2/test_median": np.nanmedian(test_r2),
                    "r2/train_argmax": np.argmax(train_r2),
                    "r2/test_argmax": np.argmax(test_r2),
                    "intrinsic_dim/last": intrinsic_dimensions[-1],
                    "intrinsic_dim/min": intrinsic_dimensions.min(),
                    "intrinsic_dim/max": intrinsic_dimensions.max(),
                    "intrinsic_dim/argmin": np.argmin(intrinsic_dimensions),
                    "intrinsic_dim/argmax": np.argmax(intrinsic_dimensions),
                }, step=epoch)

            if verbose:
                print(
                    f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                    f"Test Acc UB={test_acc_ub.max():.4f}, "
                    f"Test Acc B={test_acc_b.max():.4f}, "
                    f"Test R2={test_r2.max():.4f}",
                    flush=True
                )
        
        if chkpt_freq > 0 and epoch % chkpt_freq == 0:
            checkpoint_path = run_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)
    
    checkpoint_path = run_dir / "checkpoints" / f"checkpoint_final.pt"
    torch.save(model.state_dict(), checkpoint_path)
    np.savez(run_dir / "loss_history.npz", train_loss=np.array(train_loss_history), val_loss=np.array(val_loss_history))

    metric_history_dict = {k: np.stack([m[k] for m in metric_history]) for k in metric_history[0].keys()}
    np.savez(run_dir / "metric_history.npz", **metric_history_dict)

    if run is not None:
        run.finish()
    
    time.sleep(1.0)


if __name__ == "__main__":    
    main()