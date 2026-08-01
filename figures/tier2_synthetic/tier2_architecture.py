"""Render a detailed connection diagram of the Tier 2 disentangled stack.

Data flow (solid arrows):  X -> ASTAEncoder -> h -> MicroEncoder -> c
                           h, c -> FiLM -> h' = gamma(c)*h + beta(c)
                           c -> Adversary -> env logits

Gradient flow (dashed):  the gradient-reversal (GR) layer inside the
adversary path flips the sign so the featurizer (encoder / h_phi / FiLM)
gets pushed AWAY from making c env-predictive. The adversary's own
weights see the un-flipped gradient.

External heads from train_tier2_synthetic.DisentangledTrainer:
  - LocationHeadPerTimestep   (reads h')
  - nlos_mlp                  (reads c, replicated across T)
  - cond_head                 (reads c, auxiliary)

Output: figures/tier2_synthetic/tier2_architecture.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


# Layout
# Columns: X  ->  Encoder  ->  h  ->  MicroEnc  ->  c  ->  FiLM  ->  h'  ->  Heads
# Adversary branches off of c (after GR), feeds back to encoder via dashed.
#
# We render forward arrows in darkblue and gradient-back arrows in red (dashed).

# Colors
C_ENC     = "#1f77b4"   # blue
C_MIC     = "#17becf"   # teal
C_FILM    = "#9467bd"   # purple
C_ADV     = "#d62728"   # red
C_HEAD    = "#2ca02c"   # green
C_INPUT   = "#7f7f7f"   # gray
C_LOSS    = "#ff7f0e"   # orange


def box(ax, x, y, w, h, text, color, fc=None, fontsize=9, fontweight="normal"):
    fc = fc or color
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.6, edgecolor=color, facecolor=fc, alpha=0.18,
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color="#222")


def arrow(ax, x0, y0, x1, y1, color, label="", style="-", lw=1.6,
          label_offset=(0, 0.10), label_color=None):
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="->", mutation_scale=14,
        color=color, linewidth=lw, linestyle=style,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arr)
    if label:
        mx, my = (x0 + x1) / 2 + label_offset[0], (y0 + y1) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=8,
                color=label_color or color,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))


def main(out_path: str = "figures/tier2_synthetic/tier2_architecture.png"):
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Tier 2 — Disentangled Stack (forward + gradient flow)",
                 fontsize=15, fontweight="bold", pad=14)

    # ---------------- Nodes (x, y) ----------------
    # Inputs
    box(ax, 0.7, 8.2, 1.5, 0.55, "X  (B, T, M, 2)\nToA + AoA", C_INPUT, fc="#eee")
    box(ax, 0.7, 6.4, 1.5, 0.55, "env_id\n(0..n_envs-1)", C_INPUT, fc="#eee")
    box(ax, 0.7, 4.6, 1.5, 0.55, "cond_id\n(0/1/2)", C_INPUT, fc="#eee")
    box(ax, 0.7, 2.8, 1.5, 0.55, "NLOS mask\n(B, T, M)", C_INPUT, fc="#eee")
    box(ax, 0.7, 1.0, 1.5, 0.55, "(x, y)\n(B, T, 2)", C_INPUT, fc="#eee")

    # Encoder + micro + FiLM
    box(ax, 3.0, 7.6, 1.9, 0.7, "ASTAEncoder\n(Tier 1 / SSL init)", C_ENC, fontsize=10, fontweight="bold")
    box(ax, 3.0, 6.0, 1.9, 0.7, "h  (B, T, M, d)\nFiLM-modulated", C_ENC, fc="#dbe9f5")

    box(ax, 5.6, 8.2, 1.9, 0.7, "MicroEncoder  h_φ\nattn pool + MLP", C_MIC, fontsize=10, fontweight="bold")
    box(ax, 5.6, 6.9, 1.9, 0.7, "c  (B, micro_dim)\ncondition vector", C_MIC, fc="#d6f0ee")

    box(ax, 8.4, 7.6, 1.9, 0.7, "FiLM\nγ = 1 + γ_net(c)\nβ =  β_net(c)", C_FILM, fontsize=10, fontweight="bold")
    box(ax, 8.4, 5.9, 1.9, 0.7, "h' = γ(c)·h + β(c)\n(B, T, M, d)", C_FILM, fc="#e6dcef")

    # Adversary + GR
    box(ax, 5.6, 4.6, 1.9, 0.6, "GradReverse  GR\n(forward = I,\nbackward = −λ)", C_ADV, fc="#fde2e2", fontsize=9, fontweight="bold")
    box(ax, 8.4, 4.6, 1.9, 0.7, "Adversary  D\n2-layer MLP", C_ADV, fontsize=10, fontweight="bold")
    box(ax, 11.0, 4.6, 1.7, 0.6, "env logits\n(B, n_envs)", C_ADV, fc="#fde2e2")

    # Heads
    box(ax, 11.0, 7.6, 1.9, 0.7, "LocationHead\nPerTimestep", C_HEAD, fontsize=10, fontweight="bold")
    box(ax, 11.0, 8.9, 1.9, 0.7, "loc  (B, T, 2)", C_HEAD, fc="#dfeedd")

    box(ax, 13.2, 6.4, 1.9, 0.7, "nlos_mlp\nc -> (B, M)", C_HEAD, fontsize=10, fontweight="bold")
    box(ax, 13.2, 7.6, 1.9, 0.7, "nlos_logits\n(B, T, M)", C_HEAD, fc="#dfeedd")

    box(ax, 11.0, 5.7, 1.9, 0.7, "cond_head\nLinear(c)", C_HEAD, fontsize=10, fontweight="bold")
    box(ax, 11.0, 4.4, 1.9, 0.7, "cond_logits\n(B, 3)", C_HEAD, fc="#dfeedd")

    # Losses
    box(ax, 14.6, 8.9, 1.3, 0.6, "L_task\nMSE(loc, y)", C_LOSS, fc="#fde9d9", fontsize=9)
    box(ax, 14.6, 4.6, 1.3, 0.6, "L_adv\nCE(env_logits, env_id)", C_LOSS, fc="#fde9d9", fontsize=9)
    box(ax, 14.6, 7.6, 1.3, 0.6, "L_nlos\nBCE(nlos, mask)", C_LOSS, fc="#fde9d9", fontsize=9)
    box(ax, 14.6, 4.4, 1.3, 0.6, "L_cond\nCE(cond_logits, cond_id)", C_LOSS, fc="#fde9d9", fontsize=9)

    # ---------------- Forward (data) arrows ----------------
    arrow(ax, 1.5, 8.2, 2.05, 7.6, "black", "X")
    arrow(ax, 2.05, 7.25, 2.05, 6.35, "black")                    # encoder h  (down to "h" panel)
    arrow(ax, 2.05, 5.65, 2.05, 6.3, "black", "h", style="-",
          label_offset=(0.3, 0.0))                                  # into h box

    arrow(ax, 3.95, 6.0, 4.6, 8.2, C_ENC, "h")                     # h -> MicroEncoder
    arrow(ax, 4.65, 6.9, 7.4, 7.6, C_MIC, "c")                     # c -> FiLM
    arrow(ax, 3.95, 5.65, 7.4, 7.6, C_ENC, "h", label_offset=(0, -0.30))

    arrow(ax, 9.35, 7.6, 10.05, 7.6, C_FILM, "h'")                 # FiLM -> LocationHead
    arrow(ax, 11.95, 8.2, 11.95, 8.9, C_HEAD)                      # head -> loc
    arrow(ax, 11.95, 8.9, 13.95, 8.9, C_LOSS, "→ L_task", label_color=C_LOSS)

    arrow(ax, 9.35, 5.9, 12.3, 7.3, C_FILM, "h'")                   # FiLM -> nlos_mlp (path)
    arrow(ax, 13.2, 6.7, 13.2, 7.25, C_HEAD)                       # nlos_mlp -> logits
    arrow(ax, 13.85, 7.6, 13.95, 7.6, C_LOSS, "→ L_nlos", label_color=C_LOSS)

    # c -> Adversary via GradReverse
    arrow(ax, 6.55, 6.55, 6.55, 4.9, C_ADV, "c", label_offset=(0.3, 0))
    arrow(ax, 6.55, 4.6, 7.4, 4.6, C_ADV, "GR(c)")
    arrow(ax, 9.35, 4.6, 10.15, 4.6, C_ADV, "logits")
    arrow(ax, 11.85, 4.6, 13.95, 4.6, C_LOSS, "→ L_adv", label_color=C_LOSS)

    # c -> cond_head (auxiliary)
    arrow(ax, 6.55, 6.6, 10.05, 5.7, C_MIC, "c")
    arrow(ax, 11.95, 4.75, 11.95, 4.75, C_HEAD)
    arrow(ax, 11.95, 4.75, 13.95, 4.4, C_LOSS, "→ L_cond", label_color=C_LOSS)

    # Inputs -> losses
    arrow(ax, 1.5, 6.4, 13.95, 4.6, "gray", "env_id", style="--", lw=0.9, label_offset=(0, 0.25))
    arrow(ax, 1.5, 4.6, 13.95, 4.4, "gray", "cond_id", style="--", lw=0.9, label_offset=(0, 0.25))
    arrow(ax, 1.5, 2.8, 13.95, 7.6, "gray", "NLOS", style="--", lw=0.9, label_offset=(0, 0.25))
    arrow(ax, 1.5, 1.0, 13.95, 8.9, "gray", "(x, y)", style="--", lw=0.9, label_offset=(0, 0.25))

    # ---------------- Backward (gradient) arrows ----------------
    # Loss -> adversary (adv gets normal -grad)
    arrow(ax, 13.95, 4.85, 9.35, 4.85, C_ADV, "∇L_adv → D", style="--", lw=1.2,
          label_offset=(0, 0.30), label_color=C_ADV)
    # Through GR (flips sign)
    arrow(ax, 7.4, 4.35, 7.4, 4.35, C_ADV)
    arrow(ax, 7.4, 4.35, 6.55, 4.35, C_ADV, "−λ·∇L_adv", style="--", lw=1.2,
          label_offset=(0, 0.30), label_color=C_ADV)
    arrow(ax, 6.55, 4.35, 6.55, 6.45, C_ADV, "→ h_φ", style="--", lw=1.2,
          label_offset=(0.6, 0), label_color=C_ADV)
    # h_φ -> encoder (also gets -grad)
    arrow(ax, 4.65, 8.0, 3.95, 7.5, C_ADV, "→ ASTAEncoder", style="--", lw=1.2,
          label_offset=(0, 0.30), label_color=C_ADV)

    # Cond loss -> cond_head (positive pressure: keeps c cond-informative)
    arrow(ax, 13.95, 4.2, 11.95, 4.2, C_MIC, "∇L_cond", style="--", lw=1.2,
          label_offset=(0, 0.30), label_color=C_MIC)
    arrow(ax, 11.95, 4.2, 6.55, 4.2, C_MIC, "→ h_φ  (positive)", style="--", lw=1.2,
          label_offset=(0, 0.30), label_color=C_MIC)

    # Task + NLOS grads -> FiLM/encoder (normal +grad)
    arrow(ax, 13.95, 8.7, 11.95, 8.4, C_HEAD, "∇L_task", style="--", lw=1.2,
          label_offset=(0, 0.30), label_color=C_HEAD)
    arrow(ax, 11.95, 7.3, 9.35, 7.3, C_HEAD, "→ LocationHead/FiLM/Encoder", style="--", lw=1.2,
          label_offset=(0, 0.30), label_color=C_HEAD)
    arrow(ax, 13.95, 7.35, 13.2, 7.35, C_HEAD, "∇L_nlos", style="--", lw=1.2,
          label_offset=(0, 0.30), label_color=C_HEAD)

    # ---------------- Legend ----------------
    legend_x = 0.2
    legend_y = 0.05
    ax.text(legend_x, legend_y + 0.55, "Legend:", fontsize=9, fontweight="bold")
    ax.plot([legend_x + 0.7, legend_x + 1.4], [legend_y + 0.55, legend_y + 0.55],
            color="black", lw=1.6, label="forward (data)")
    ax.text(legend_x + 1.5, legend_y + 0.55, "forward", fontsize=8, va="center")
    ax.plot([legend_x + 0.7, legend_x + 1.4], [legend_y + 0.30, legend_y + 0.30],
            color=C_ADV, lw=1.6, linestyle="--", label="backward (grad)")
    ax.text(legend_x + 1.5, legend_y + 0.30, "grad (reversed via GR)", fontsize=8, va="center")
    ax.plot([legend_x + 0.7, legend_x + 1.4], [legend_y + 0.05, legend_y + 0.05],
            color="gray", lw=0.9, linestyle="--")
    ax.text(legend_x + 1.5, legend_y + 0.05, "label flow (target side)", fontsize=8, va="center")

    # Sidebar: key idea
    note = (
        "Key:  c is the *disentangled* micro vector.\n"
        "  • L_adv pushes c AWAY from env (via GR sign flip)\n"
        "  • L_cond pulls c TOWARD cond   (positive pressure)\n"
        "  • L_task + L_nlos shape h'  (modulated features)\n"
        "  • Net effect: c = micro-only,  h' = micro-aware features"
    )
    ax.text(0.4, 5.6, note, fontsize=9, va="center", ha="left",
            bbox=dict(facecolor="white", edgecolor="#aaa", boxstyle="round,pad=0.5"))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
