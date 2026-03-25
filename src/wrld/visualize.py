"""Visualization utilities: loss curves, reconstruction, rollout, latent PCA."""
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


def plot_loss_curves(history: dict, save_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training Loss Curves", fontsize=14)

    pairs = [
        ("Reconstruction Loss", "train_recon", "val_recon"),
        ("Dynamics Loss", "train_dynamics", "val_dynamics"),
        ("KL Divergence", "train_kl", "val_kl"),
        ("Total Loss", "train_total", "val_total"),
    ]

    for ax, (title, train_key, val_key) in zip(axes.flat, pairs):
        epochs = range(1, len(history[train_key]) + 1)
        ax.semilogy(epochs, history[train_key], label="train")
        ax.semilogy(epochs, history[val_key], label="val", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (log scale)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curves to {save_path}")


def plot_reconstruction(model, val_data: dict, device: torch.device, save_path: Path) -> None:
    """Plot true vs reconstructed for the 4 worst-case validation sequences (highest MSE)."""
    model.eval()
    all_obs = val_data["observations"].to(device)  # (N, T, 2)
    N, T, obs_dim = all_obs.shape

    with torch.no_grad():
        obs_flat = all_obs.reshape(N * T, obs_dim)
        mu, _ = model.encoder(obs_flat)
        recon_flat = model.decoder(mu)
        recon_all = recon_flat.reshape(N, T, obs_dim)

        # Per-sequence MSE
        mse_per_seq = ((recon_all - all_obs) ** 2).mean(dim=(1, 2))  # (N,)
        worst_idxs = torch.argsort(mse_per_seq, descending=True)[:4]

    obs_np = all_obs.cpu().numpy()
    recon_np = recon_all.cpu().numpy()
    mse_np = mse_per_seq.cpu().numpy()

    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    fig.suptitle("Reconstruction — 4 Worst Sequences (highest MSE)", fontsize=14)

    for row, idx in enumerate(worst_idxs):
        idx = idx.item()
        t = np.arange(T)
        ax_pos = axes[row, 0]
        ax_vel = axes[row, 1]

        ax_pos.plot(t, obs_np[idx, :, 0], label="ground truth", color="steelblue")
        ax_pos.plot(t, recon_np[idx, :, 0], label="reconstruction", color="tomato", linestyle="--")
        ax_pos.set_title(f"Seq {idx} — Position  (MSE={mse_np[idx]:.4f})")
        ax_pos.set_xlabel("Timestep")
        ax_pos.set_ylabel("Position")
        ax_pos.legend(fontsize=8)
        ax_pos.grid(True, alpha=0.3)

        ax_vel.plot(t, obs_np[idx, :, 1], label="ground truth", color="steelblue")
        ax_vel.plot(t, recon_np[idx, :, 1], label="reconstruction", color="tomato", linestyle="--")
        ax_vel.set_title(f"Seq {idx} — Velocity  (MSE={mse_np[idx]:.4f})")
        ax_vel.set_xlabel("Timestep")
        ax_vel.set_ylabel("Velocity")
        ax_vel.legend(fontsize=8)
        ax_vel.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reconstruction plot to {save_path}")


def plot_rollout(model, val_data: dict, device: torch.device, save_path: Path) -> None:
    """Open-loop rollout for the 4 worst-case validation sequences (highest MSE)."""
    model.eval()
    all_obs = val_data["observations"].to(device)      # (N, T, 2)
    all_actions = val_data["actions"].to(device)        # (N, T-1, 1)
    N, T, obs_dim = all_obs.shape

    with torch.no_grad():
        # Roll out all sequences to find worst cases
        mu_0, _ = model.encoder(all_obs[:, 0, :])
        z = mu_0  # (N, 30)

        preds = [model.decoder(z)]  # list of (N, 2)
        for t in range(T - 1):
            z = model.dynamics(z, all_actions[:, t, :])
            preds.append(model.decoder(z))

        predicted = torch.stack(preds, dim=1)  # (N, T, 2)

        # Per-sequence MSE
        mse_per_seq = ((predicted - all_obs) ** 2).mean(dim=(1, 2))  # (N,)
        worst_idxs = torch.argsort(mse_per_seq, descending=True)[:4]

    obs_np = all_obs.cpu().numpy()
    pred_np = predicted.cpu().numpy()
    mse_np = mse_per_seq.cpu().numpy()

    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    fig.suptitle("Open-Loop Rollout — 4 Worst Sequences (highest MSE)", fontsize=14)

    for row, idx in enumerate(worst_idxs):
        idx = idx.item()
        t = np.arange(T)
        ax_pos = axes[row, 0]
        ax_vel = axes[row, 1]

        ax_pos.plot(t, obs_np[idx, :, 0], label="ground truth", color="steelblue")
        ax_pos.plot(t, pred_np[idx, :, 0], label="rollout", color="darkorange", linestyle="--")
        ax_pos.set_title(f"Seq {idx} — Position  (MSE={mse_np[idx]:.4f})")
        ax_pos.set_xlabel("Timestep")
        ax_pos.set_ylabel("Position")
        ax_pos.legend(fontsize=8)
        ax_pos.grid(True, alpha=0.3)

        ax_vel.plot(t, obs_np[idx, :, 1], label="ground truth", color="steelblue")
        ax_vel.plot(t, pred_np[idx, :, 1], label="rollout", color="darkorange", linestyle="--")
        ax_vel.set_title(f"Seq {idx} — Velocity  (MSE={mse_np[idx]:.4f})")
        ax_vel.set_xlabel("Timestep")
        ax_vel.set_ylabel("Velocity")
        ax_vel.legend(fontsize=8)
        ax_vel.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved rollout plot to {save_path}")


def plot_latent_pca(model, device: torch.device, save_path: Path) -> None:
    """PCA of latent encodings over a sweep of positions."""
    from sklearn.decomposition import PCA

    positions = np.linspace(-3.0, 3.0, 200).astype(np.float32)
    # sweep position with zero velocity
    obs_tensor = torch.from_numpy(
        np.stack([positions, np.zeros_like(positions)], axis=1)
    ).to(device)

    model.eval()
    with torch.no_grad():
        mu, _ = model.encoder(obs_tensor)
        z = mu.cpu().numpy()

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=positions, cmap="RdBu_r", s=20)
    plt.colorbar(sc, ax=ax, label="Position")
    ax.set_title("Latent Space PCA (colored by position, zero velocity)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved latent PCA plot to {save_path}")
