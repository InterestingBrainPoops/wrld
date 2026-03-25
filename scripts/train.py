"""Train the world model, save checkpoints and plots."""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wrld.data import generate_sequences, save_data, load_data, make_dataloader
from wrld.models import WorldModel
from wrld.train import train
from wrld.visualize import (
    plot_loss_curves,
    plot_reconstruction,
    plot_rollout,
    plot_latent_pca,
)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "outputs" / "data"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"
PLOT_DIR = ROOT / "outputs" / "plots"


def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate data if not present
    train_path = DATA_DIR / "train.pt"
    val_path = DATA_DIR / "val.pt"
    if not train_path.exists() or not val_path.exists():
        print("Data not found — generating...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_data = generate_sequences(num_sequences=2000, seq_len=64, seed=42)
        val_data = generate_sequences(num_sequences=200, seq_len=64, seed=123)
        save_data(train_data, train_path)
        save_data(val_data, val_path)
    else:
        print("Loading existing data...")
        train_data = load_data(train_path)
        val_data = load_data(val_path)

    train_loader = make_dataloader(train_data, batch_size=128, shuffle=True)
    val_loader = make_dataloader(val_data, batch_size=128, shuffle=False)

    # Build model
    model = WorldModel()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Train
    print("Starting training...")
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=200,
        lr=1e-3,
        kl_weight=0.001,
        kl_warmup_epochs=40,
        device=device,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_every=20,
    )

    # Save final model
    final_path = CHECKPOINT_DIR / "final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")

    # Plots
    model = model.to(device)
    plot_loss_curves(history, PLOT_DIR / "loss_curves.png")
    plot_reconstruction(model, val_data, device, PLOT_DIR / "reconstruction.png")
    plot_rollout(model, val_data, device, PLOT_DIR / "rollout.png")

    try:
        plot_latent_pca(model, device, PLOT_DIR / "latent_pca.png")
    except ImportError:
        print("sklearn not installed — skipping latent PCA plot")

    print("\nAll done. Outputs saved to outputs/")


if __name__ == "__main__":
    main()
