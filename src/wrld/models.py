"""VAE-style world model: Encoder -> Latent Dynamics -> Decoder.

Dimensions:
  observation: 2D (position, velocity)
  action:      1D (force)
  latent:      30D
"""
import torch
import torch.nn as nn

LATENT_DIM = 16
OBS_DIM = 2
ACTION_DIM = 1


class Encoder(nn.Module):
    """Maps 2D observation (position, velocity) to 30D latent distribution (mu, log_var)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            nn.Linear(64, LATENT_DIM * 2),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mu, log_var = h.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, -10.0, 10.0)
        return mu, log_var

class DynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        # A: state transition — initialize to slightly less than identity
        # so state decays slowly rather than exploding
        self.A = nn.Linear(LATENT_DIM, LATENT_DIM, bias=False)
        self.B = nn.Linear(ACTION_DIM, LATENT_DIM, bias=True)

        # nonlinearity between SSM and output
        self.out = nn.Sequential(
            nn.LayerNorm(LATENT_DIM),
            nn.ReLU(),
            nn.Linear(LATENT_DIM, LATENT_DIM),
        )

        # initialize A to be stable
        # small identity-like init: eigenvalues near but below 1
        nn.init.eye_(self.A.weight)
        self.A.weight.data *= 0.9

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self.A(z) + self.B(action)   # core SSM step
        h = self.out(h)                   # nonlinear refinement
        return z + h                      # residual — no scaling needed


class Decoder(nn.Module):
    """Maps 30D latent to 2D observation (position, velocity)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, OBS_DIM),
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
            obs_seq:    (B, T, OBS_DIM)
            action_seq: (B, T-1, ACTION_DIM)
        Returns dict with keys: mu, log_var, obs_recon, obs_target,
                                rollout_preds, rollout_targets
        """
        B, T, _ = obs_seq.shape

        # Encode all timesteps at once (for reconstruction + KL)
        obs_flat = obs_seq.reshape(B * T, OBS_DIM)
        mu_flat, log_var_flat = self.encoder(obs_flat)
        z_flat = self._reparameterize(mu_flat, log_var_flat)

        # Decode all for reconstruction loss
        obs_recon_flat = self.decoder(z_flat)

        # Full rollout: start from mu_0, iterate dynamics autoregressively
        mu_seq = mu_flat.reshape(B, T, LATENT_DIM)
        z = mu_seq[:, 0, :]  # seed with encoder mean at t=0
        z_preds = []
        for t in range(T - 1):
            z = self.dynamics(z, action_seq[:, t, :])
            z_preds.append(z)

        # Batch-decode all steps in one forward pass instead of T-1 separate calls
        z_preds = torch.stack(z_preds, dim=1)                              # (B, T-1, LATENT_DIM)
        rollout_preds = self.decoder(z_preds.reshape(B * (T - 1), LATENT_DIM)).reshape(B, T - 1, OBS_DIM)
        rollout_targets = obs_seq[:, 1:, :]

        return {
            "mu": mu_flat,
            "log_var": log_var_flat,
            "obs_recon": obs_recon_flat,
            "obs_target": obs_flat,
            "rollout_preds": rollout_preds,
            "rollout_targets": rollout_targets,
        }
