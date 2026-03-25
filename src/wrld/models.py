"""VAE-style world model: Encoder -> Latent Dynamics -> Decoder.

Dimensions:
  observation: 2D (position, velocity)
  action:      1D (force)
  latent:      30D
"""
import torch
import torch.nn as nn

LATENT_DIM = 30
OBS_DIM = 2
ACTION_DIM = 1


class Encoder(nn.Module):
    """Maps 2D observation (position, velocity) to 30D latent distribution (mu, log_var)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, LATENT_DIM * 2),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mu, log_var = h.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, -10.0, 10.0)
        return mu, log_var


class DynamicsModel(nn.Module):
    """Predicts next latent from current latent + action (residual)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM + ACTION_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, LATENT_DIM),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z, action], dim=-1)
        return z + self.net(inp)  # residual: predict delta


class Decoder(nn.Module):
    """Maps 30D latent to 2D observation (position, velocity)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, OBS_DIM),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class WorldModel(nn.Module):
    """Wraps encoder, dynamics, and decoder. Handles reparameterization."""

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.dynamics = DynamicsModel()
        self.decoder = Decoder()

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(obs)
        z = self._reparameterize(mu, log_var)
        return mu, log_var, z

    def _reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, obs_seq: torch.Tensor, action_seq: torch.Tensor) -> dict:
        """
        Args:
            obs_seq:    (B, T, 1)
            action_seq: (B, T-1, 1)
        Returns dict with keys: mu, log_var, obs_recon, z_next_pred, z_next_target
        """
        B, T, _ = obs_seq.shape

        # Encode all timesteps at once
        obs_flat = obs_seq.reshape(B * T, OBS_DIM)
        mu_flat, log_var_flat = self.encoder(obs_flat)
        z_flat = self._reparameterize(mu_flat, log_var_flat)

        # Decode all for reconstruction
        obs_recon_flat = self.decoder(z_flat)

        # Dynamics predictions
        z_seq = z_flat.reshape(B, T, LATENT_DIM)
        z_current = z_seq[:, :-1, :].reshape(B * (T - 1), LATENT_DIM)
        z_next_target = z_seq[:, 1:, :].detach().reshape(B * (T - 1), LATENT_DIM)
        actions_flat = action_seq.reshape(B * (T - 1), ACTION_DIM)
        z_next_pred = self.dynamics(z_current, actions_flat)

        return {
            "mu": mu_flat,
            "log_var": log_var_flat,
            "obs_recon": obs_recon_flat,
            "obs_target": obs_flat,
            "z_next_pred": z_next_pred,
            "z_next_target": z_next_target,
        }
