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

    dynamics_loss = F.mse_loss(out["z_next_pred"], out["z_next_target"])

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


def train(
    model: WorldModel,
    train_loader,
    val_loader,
    num_epochs: int = 200,
    lr: float = 1e-3,
    kl_weight: float = 0.001,
    kl_warmup_epochs: int = 40,
    device: torch.device = None,
    checkpoint_dir=None,
    checkpoint_every: int = 20,
) -> dict:
    """Full training loop. Returns loss history dict."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {
        "train_total": [], "train_recon": [], "train_dynamics": [], "train_kl": [],
        "val_total": [], "val_recon": [], "val_dynamics": [], "val_kl": [],
        "epoch_seconds": [],
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

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"train total={history['train_total'][-1]:.4f} "
                f"recon={history['train_recon'][-1]:.4f} "
                f"dyn={history['train_dynamics'][-1]:.4f} "
                f"kl={history['train_kl'][-1]:.4f} | "
                f"val total={history['val_total'][-1]:.4f} | "
                f"epoch time={epoch_seconds:.2f}s"
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
