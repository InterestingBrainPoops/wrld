"""Training loop and loss computation for the world model."""
from time import perf_counter

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from wrld.models import WorldModel


def compute_losses(
    model: WorldModel,
    obs_seq: torch.Tensor,
    action_seq: torch.Tensor,
    kl_weight: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single forward pass computing reconstruction, dynamics, and KL losses."""
    out = model(obs_seq, action_seq)

    recon_loss = F.mse_loss(out["obs_recon"], out["obs_target"])

    dynamics_loss = F.mse_loss(out["rollout_preds"], out["rollout_targets"])

    kl_loss = -0.5 * torch.mean(
        1 + out["log_var"] - out["mu"].pow(2) - out["log_var"].exp()
    )

    total = recon_loss + dynamics_loss + kl_weight * kl_loss
    return total, recon_loss, dynamics_loss, kl_loss


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "synchronize")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.synchronize()


def _rollout_mse(model: WorldModel, val_obs: torch.Tensor, val_actions: torch.Tensor) -> torch.Tensor:
    """Returns per-sequence rollout MSE over the full val set. Shape: (N,)"""
    N, T, _ = val_obs.shape
    with torch.no_grad():
        mu_0, _ = model.encoder(val_obs[:, 0, :])
        z = mu_0
        preds = [model.decoder(z)]
        for t in range(T - 1):
            z = model.dynamics(z, val_actions[:, t, :])
            preds.append(model.decoder(z))
        predicted = torch.stack(preds, dim=1)  # (N, T, 2)
        return ((predicted - val_obs) ** 2).mean(dim=(1, 2))  # (N,)


def train(
    model: WorldModel,
    train_loader,
    val_loader,
    val_data: dict = None,
    num_epochs: int = 200,
    lr: float = 1e-3,
    kl_weight: float = 0.001,
    kl_warmup_epochs: int = 40,
    device: torch.device = None,
    checkpoint_dir=None,
    checkpoint_every: int = 20,
    rollout_every: int = 10,
) -> dict:
    """Full training loop. Returns loss history dict."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Pre-move val tensors to device once if provided
    val_obs = val_actions = None
    if val_data is not None:
        val_obs = val_data["observations"].to(device)
        val_actions = val_data["actions"].to(device)

    history = {
        "train_total": [], "train_recon": [], "train_dynamics": [], "train_kl": [],
        "val_total": [], "val_recon": [], "val_dynamics": [], "val_kl": [],
        "epoch_seconds": [],
        "rollout_epochs": [], "rollout_mean": [], "rollout_std": [],
        "rollout_min": [], "rollout_max": [],
    }
    _synchronize_device(device)
    training_start = perf_counter()

    for epoch in range(num_epochs):
        _synchronize_device(device)
        epoch_start = perf_counter()

        # KL warmup: linearly increase from 0 to kl_weight
        current_kl_weight = kl_weight * min(1.0, epoch / max(kl_warmup_epochs, 1))

        model.train()
        sums = {"total": 0.0, "recon": 0.0, "dynamics": 0.0, "kl": 0.0}
        n_batches = 0

        for obs_batch, action_batch in train_loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)

            total, recon, dyn, kl = compute_losses(
                model, obs_batch, action_batch, kl_weight=current_kl_weight
            )

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            sums["total"] += total.item()
            sums["recon"] += recon.item()
            sums["dynamics"] += dyn.item()
            sums["kl"] += kl.item()
            n_batches += 1

        scheduler.step()

        for k in sums:
            history[f"train_{k}"].append(sums[k] / n_batches)

        # Validation
        model.eval()
        val_sums = {"total": 0.0, "recon": 0.0, "dynamics": 0.0, "kl": 0.0}
        n_val = 0
        with torch.no_grad():
            for obs_batch, action_batch in val_loader:
                obs_batch = obs_batch.to(device)
                action_batch = action_batch.to(device)
                total, recon, dyn, kl = compute_losses(
                    model, obs_batch, action_batch, kl_weight=current_kl_weight
                )
                val_sums["total"] += total.item()
                val_sums["recon"] += recon.item()
                val_sums["dynamics"] += dyn.item()
                val_sums["kl"] += kl.item()
                n_val += 1

        for k in val_sums:
            history[f"val_{k}"].append(val_sums[k] / n_val)

        _synchronize_device(device)
        epoch_seconds = perf_counter() - epoch_start
        history["epoch_seconds"].append(epoch_seconds)

        if (epoch + 1) % rollout_every == 0:
            rollout_str = ""
            if val_obs is not None:
                model.eval()
                mse = _rollout_mse(model, val_obs, val_actions)
                r_mean, r_std = mse.mean().item(), mse.std().item()
                r_min, r_max = mse.min().item(), mse.max().item()
                history["rollout_epochs"].append(epoch + 1)
                history["rollout_mean"].append(r_mean)
                history["rollout_std"].append(r_std)
                history["rollout_min"].append(r_min)
                history["rollout_max"].append(r_max)
                rollout_str = (
                    f" | rollout mse mean={r_mean:.4f} "
                    f"std={r_std:.4f} min={r_min:.4f} max={r_max:.4f}"
                )
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"train total={history['train_total'][-1]:.4f} "
                f"recon={history['train_recon'][-1]:.4f} "
                f"dyn={history['train_dynamics'][-1]:.4f} "
                f"kl={history['train_kl'][-1]:.4f} | "
                f"val total={history['val_total'][-1]:.4f} | "
                f"epoch time={epoch_seconds:.2f}s"
                + rollout_str
            )

        if checkpoint_dir is not None and (epoch + 1) % checkpoint_every == 0:
            from pathlib import Path
            ckpt_path = Path(checkpoint_dir) / f"epoch_{epoch+1:04d}.pt"
            torch.save(model.state_dict(), ckpt_path)

    _synchronize_device(device)
    total_train_seconds = perf_counter() - training_start
    history["total_train_seconds"] = total_train_seconds
    history["avg_epoch_seconds"] = total_train_seconds / max(num_epochs, 1)

    return history
