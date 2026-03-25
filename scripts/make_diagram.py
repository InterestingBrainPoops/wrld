"""Generate a clean architecture diagram using matplotlib."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Styling ──────────────────────────────────────────────
BG = "#0f172a"
BOX_BG = "#1e293b"
TEXT = "#e2e8f0"
SUBTEXT = "#94a3b8"
BORDER_ENC = "#3b82f6"
BORDER_DYN = "#f97316"
BORDER_DEC = "#22c55e"
BORDER_REPARAM = "#a78bfa"
BORDER_INPUT = "#64748b"
BORDER_LOSS = "#ef4444"
BORDER_ACTION = "#eab308"

W = 20   # canvas width
H = 16   # canvas height


def box(ax, cx, cy, w, h, label, sublabel, border_color, bg=BOX_BG, fontsize=13, sublabel2=None):
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=bg, edgecolor=border_color, linewidth=2,
    )
    ax.add_patch(rect)
    if sublabel2:
        ax.text(cx, cy + 0.35, label, ha="center", va="center",
                color=TEXT, fontsize=fontsize, fontweight="bold", family="monospace")
        ax.text(cx, cy - 0.2, sublabel, ha="center", va="center",
                color=SUBTEXT, fontsize=10, family="monospace")
        ax.text(cx, cy - 0.65, sublabel2, ha="center", va="center",
                color=SUBTEXT, fontsize=10, family="monospace")
    elif sublabel:
        ax.text(cx, cy + 0.2, label, ha="center", va="center",
                color=TEXT, fontsize=fontsize, fontweight="bold", family="monospace")
        ax.text(cx, cy - 0.3, sublabel, ha="center", va="center",
                color=SUBTEXT, fontsize=10, family="monospace")
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                color=TEXT, fontsize=fontsize, fontweight="bold", family="monospace")


def arr(ax, x1, y1, x2, y2, color="#64748b", lw=1.8, ls="-", cs="arc3,rad=0"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=16,
        color=color, linewidth=lw, linestyle=ls,
        connectionstyle=cs,
    )
    ax.add_patch(a)


def label(ax, x, y, text, color=SUBTEXT, fontsize=9):
    ax.text(x, y, text, ha="center", va="center",
            color=color, fontsize=fontsize, family="monospace",
            bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="none", alpha=0.9))


# ── Canvas ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(24, 18), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(-0.5, W + 0.5)
ax.set_ylim(-0.5, H + 0.5)
ax.set_aspect("equal")
ax.axis("off")

# Title
ax.text(W/2, H - 0.1, "World Model Architecture", ha="center", va="top",
        color=TEXT, fontsize=20, fontweight="bold", family="monospace")
ax.text(W/2, H - 0.7, "VAE encoder  /  latent dynamics  /  decoder", ha="center", va="top",
        color=SUBTEXT, fontsize=13, family="monospace")

# Timestep loop outline
loop = FancyBboxPatch(
    (0.3, 1.5), W - 0.6, H - 2.8,
    boxstyle="round,pad=0.1",
    facecolor="none", edgecolor="#334155", linewidth=1.5, linestyle="--",
)
ax.add_patch(loop)
ax.text(0.7, H - 1.5, "for t = 0 ... T-1", color="#475569", fontsize=11, family="monospace")

# ═══════════════════════════════════════════════════════════
# Row 1 (y=13): obs_t -> Encoder -> μ/logvar -> Reparam -> z_t -> Decoder -> obs_recon
# ═══════════════════════════════════════════════════════════
y1 = 12.5

# obs_t
box(ax, 1.5, y1, 2.0, 1.4, "obs_t", "[x, v]  (2,)", BORDER_INPUT)

# Encoder
box(ax, 5.0, y1, 2.8, 1.8, "Encoder", "2 -> 64 -> 128 -> 128 -> 60", BORDER_ENC, sublabel2="ReLU activations")

# obs_t -> Encoder
arr(ax, 2.55, y1, 3.55, y1, BORDER_ENC)

# μ
box(ax, 8.3, y1 + 1.0, 1.6, 1.0, "μ", "(30,)", BORDER_ENC, fontsize=12)
# log σ²
box(ax, 8.3, y1 - 1.0, 1.6, 1.0, "log σ²", "(30,)", BORDER_ENC, fontsize=12)

# Encoder -> mu, log_var
arr(ax, 6.45, y1 + 0.3, 7.45, y1 + 1.0, BORDER_ENC, cs="arc3,rad=-0.1")
arr(ax, 6.45, y1 - 0.3, 7.45, y1 - 1.0, BORDER_ENC, cs="arc3,rad=0.1")

# Reparameterize
box(ax, 11.0, y1, 2.4, 1.6, "Reparam", "z = μ + σ·ε", BORDER_REPARAM)

# mu, log_var -> reparam
arr(ax, 9.15, y1 + 1.0, 9.75, y1 + 0.3, BORDER_REPARAM, cs="arc3,rad=-0.1")
arr(ax, 9.15, y1 - 1.0, 9.75, y1 - 0.3, BORDER_REPARAM, cs="arc3,rad=0.1")

# z_t
box(ax, 14.0, y1, 1.6, 1.2, "z_t", "(30,)", BORDER_REPARAM, fontsize=12)

# reparam -> z_t
arr(ax, 12.25, y1, 13.15, y1, BORDER_REPARAM)
label(ax, 12.7, y1 + 0.4, "sample", BORDER_REPARAM)

# Decoder
box(ax, 17.0, y1, 2.8, 1.8, "Decoder", "30 -> 128 -> 64 -> 2", BORDER_DEC, sublabel2="ReLU, no final act.")

# z_t -> decoder
arr(ax, 14.85, y1, 15.55, y1, BORDER_DEC)

# obs_recon
box(ax, 17.0, y1 - 2.5, 2.2, 1.2, "obs_recon", "[x, v]  (2,)", BORDER_DEC, fontsize=12)

# decoder -> obs_recon
arr(ax, 17.0, y1 - 0.95, 17.0, y1 - 1.85, BORDER_DEC)

# ═══════════════════════════════════════════════════════════
# Row 2 (y=8): action_t -> concat(z_t, a_t) -> Dynamics -> + -> z_{t+1}
# ═══════════════════════════════════════════════════════════
y2 = 8.0

# action_t
box(ax, 1.5, y2, 2.0, 1.4, "action_t", "F  (1,)", BORDER_ACTION)

# concat
box(ax, 5.5, y2, 1.8, 1.2, "cat", "[z_t, a_t]  (31,)", "#64748b", fontsize=12)

# action -> concat
arr(ax, 2.55, y2, 4.55, y2, BORDER_ACTION)

# z_t -> concat (route left side to avoid crossing dynamics)
arr(ax, 14.0, y1 - 0.65, 14.0, y2 + 2.2, "#64748b")
arr(ax, 14.0, y2 + 2.2, 3.5, y2 + 2.2, "#64748b")
arr(ax, 3.5, y2 + 2.2, 3.5, y2 + 0.1, "#64748b")
arr(ax, 3.5, y2 + 0.1, 4.55, y2 + 0.1, "#64748b")
label(ax, 8.5, y2 + 2.5, "z_t  (30,)", BORDER_REPARAM)

# Dynamics
box(ax, 9.5, y2, 3.2, 1.8, "Dynamics", "31 -> 128 -> 128 -> 128 -> 30", BORDER_DYN, sublabel2="ReLU activations")

# concat -> dynamics
arr(ax, 6.45, y2, 7.85, y2, BORDER_DYN)

# + (residual add)
box(ax, 13.0, y2, 1.0, 1.0, "+", None, BORDER_DYN, fontsize=18)

# dynamics -> +
arr(ax, 11.15, y2, 12.45, y2, BORDER_DYN)
label(ax, 11.8, y2 + 0.4, "Δz", BORDER_DYN)

# z_t skip connection (residual) -> +
arr(ax, 14.0, y1 - 0.65, 13.3, y2 + 0.55, BORDER_REPARAM, ls="--", cs="arc3,rad=-0.2")
label(ax, 14.5, y2 + 1.5, "skip (residual)", SUBTEXT, fontsize=9)

# z_{t+1}
box(ax, 15.5, y2, 2.0, 1.2, "z_{t+1}", "(30,)", BORDER_DYN, fontsize=12)

# + -> z_{t+1}
arr(ax, 13.55, y2, 14.45, y2, BORDER_DYN)

# ═══════════════════════════════════════════════════════════
# Recurrence loop: z_{t+1} -> back to dynamics input at next timestep
# ═══════════════════════════════════════════════════════════
y_loop = 5.5
# down from z_{t+1}
arr(ax, 15.5, y2 - 0.65, 15.5, y_loop + 0.2, BORDER_DYN, ls="--")
# across the bottom
arr(ax, 15.5, y_loop + 0.2, 5.5, y_loop + 0.2, BORDER_DYN, ls="--")
# up to concat area (this represents next timestep)
arr(ax, 5.5, y_loop + 0.2, 5.5, y2 - 0.65, BORDER_DYN, ls="--")
label(ax, 10.5, y_loop - 0.15, "rollout loop:  z_{t+1}  becomes  z_t  at next step", BORDER_DYN, fontsize=10)

# z_{t+1} can also be decoded
arr(ax, 15.5, y2 + 0.65, 16.3, y1 - 2.5, BORDER_DEC, ls="--", cs="arc3,rad=-0.2")
label(ax, 16.5, y2 + 1.5, "decode at\nrollout", BORDER_DEC, fontsize=9)

# ═══════════════════════════════════════════════════════════
# Row 3 (y=3): Losses
# ═══════════════════════════════════════════════════════════
y3 = 3.0

box(ax, 3.5, y3, 3.8, 1.4, "Recon Loss", "MSE(obs_recon, obs_t)", BORDER_LOSS)
box(ax, 10.0, y3, 4.5, 1.4, "Dynamics Loss", "MSE(z_pred, z_{t+1}.detach())", BORDER_LOSS)
box(ax, 16.5, y3, 3.4, 1.4, "KL Loss  (β=.001)", "KL( q(z|x) || N(0,I) )", BORDER_LOSS)

# Loss arrows (dotted)
# obs_recon -> recon loss
arr(ax, 17.0, y1 - 3.15, 4.5, y3 + 0.75, BORDER_LOSS, ls=":", lw=1.3, cs="arc3,rad=0.2")
# obs_t -> recon loss
arr(ax, 1.5, y1 - 0.75, 2.5, y3 + 0.75, BORDER_LOSS, ls=":", lw=1.3, cs="arc3,rad=0")

# dynamics -> dynamics loss
arr(ax, 9.5, y2 - 0.95, 10.0, y3 + 0.75, BORDER_LOSS, ls=":", lw=1.3)

# encoder dist -> KL loss
arr(ax, 11.0, y1 - 0.85, 16.5, y3 + 0.75, BORDER_LOSS, ls=":", lw=1.3, cs="arc3,rad=-0.25")

# ── Total loss note ──────────────────────────────────────
ax.text(W/2, 1.5, "total = recon + dynamics + β · KL          (single optimizer, single backward pass)",
        ha="center", va="center", color=SUBTEXT, fontsize=12, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#1e293b", ec="#334155", lw=1.2))

# ── Save ─────────────────────────────────────────────────
out_svg = Path(__file__).parent.parent / "outputs" / "architecture.svg"
out_png = Path(__file__).parent.parent / "outputs" / "architecture.png"
fig.savefig(out_svg, format="svg", facecolor=BG, bbox_inches="tight", pad_inches=0.4)
fig.savefig(out_png, format="png", facecolor=BG, bbox_inches="tight", pad_inches=0.4, dpi=150)
plt.close()
print(f"Saved to {out_svg}")
print(f"Saved to {out_png}")
