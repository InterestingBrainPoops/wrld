"""Generate training and validation data for the spring-mass-damper world model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wrld.data import generate_sequences, save_data

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "data"


def main():
    print("Generating training data (2000 sequences)...")
    train_data = generate_sequences(num_sequences=2000, seq_len=64, seed=42)
    save_data(train_data, OUTPUT_DIR / "train.pt")

    print("Generating validation data (200 sequences)...")
    val_data = generate_sequences(num_sequences=200, seq_len=64, seed=123)
    save_data(val_data, OUTPUT_DIR / "val.pt")

    print("Done.")


if __name__ == "__main__":
    main()
