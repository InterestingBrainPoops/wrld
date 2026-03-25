"""Data generation and dataset utilities for the spring-mass-damper world model."""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from wrld.env import SpringMassDamperEnv


def _generate_force_profile(seq_len: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random force sequence using one of four profiles."""
    profile = rng.choice(["step", "sine", "impulse", "smooth"], p=[0.4, 0.2, 0.2, 0.2])

    if profile == "step":
        forces = np.zeros(seq_len)
        i = 0
        while i < seq_len:
            f = rng.uniform(-3.0, 3.0)
            hold = int(rng.integers(5, 21))
            forces[i:i + hold] = f
            i += hold

    elif profile == "sine":
        t = np.arange(seq_len)
        A = rng.uniform(0.5, 3.0)
        omega = rng.uniform(0.5, 3.0)
        phi = rng.uniform(0.0, 2 * np.pi)
        forces = A * np.sin(omega * t + phi)

    elif profile == "impulse":
        forces = np.zeros(seq_len)
        num_impulses = int(rng.integers(1, 6))
        for _ in range(num_impulses):
            idx = int(rng.integers(0, seq_len))
            dur = int(rng.integers(1, 4))
            mag = rng.uniform(-5.0, 5.0)
            forces[idx:idx + dur] = mag

    else:  # smooth
        noise = rng.standard_normal(seq_len)
        alpha = rng.uniform(0.05, 0.4)
        forces = np.zeros(seq_len)
        forces[0] = noise[0]
        for i in range(1, seq_len):
            forces[i] = alpha * noise[i] + (1 - alpha) * forces[i - 1]
        forces *= rng.uniform(1.0, 3.0)

    return forces.astype(np.float32)


def generate_sequences(
    num_sequences: int,
    seq_len: int = 64,
    env_kwargs: dict = None,
    seed: int = 0,
) -> dict:
    """Generate observation and action sequences from the spring-mass-damper env."""
    if env_kwargs is None:
        env_kwargs = {}

    rng = np.random.default_rng(seed)
    env = SpringMassDamperEnv(**env_kwargs)

    observations = np.zeros((num_sequences, seq_len, 2), dtype=np.float32)
    actions = np.zeros((num_sequences, seq_len - 1, 1), dtype=np.float32)

    for i in range(num_sequences):
        forces = _generate_force_profile(seq_len, rng)
        x0 = float(rng.uniform(-2.0, 2.0))
        v0 = float(rng.uniform(-2.0, 2.0))

        obs = env.reset(x0=x0, v0=v0)
        observations[i, 0, :] = obs  # [x, v]

        for t in range(seq_len - 1):
            obs = env.step(forces[t])
            observations[i, t + 1, :] = obs  # [x, v]
            actions[i, t, 0] = forces[t]

    return {
        "observations": torch.from_numpy(observations),
        "actions": torch.from_numpy(actions),
        "metadata": {
            "seq_len": seq_len,
            "num_sequences": num_sequences,
            **env_kwargs,
        },
    }


def save_data(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    print(f"Saved {data['metadata']['num_sequences']} sequences to {path}")


def load_data(path: Path) -> dict:
    return torch.load(path, weights_only=False)


class SequenceDataset(Dataset):
    def __init__(self, observations: torch.Tensor, actions: torch.Tensor):
        self.observations = observations
        self.actions = actions

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


def make_dataloader(data: dict, batch_size: int = 128, shuffle: bool = True) -> DataLoader:
    dataset = SequenceDataset(data["observations"], data["actions"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=0,
    )
