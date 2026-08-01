"""Render a detailed connection diagram of the Tier 1 MAML stack.

Layers (bottom -> top, since MAML sits on top of Tier 0 SSL):
  Tier 0:  ASTA encoder (shared, SSL-pretrained) + SSL head
  Tier 1:  MAMLCore = ASTAEncoder + LocationHead
           MAML outer loop samples N envs, runs K inner steps on support,
           evaluates on query, backprops the *initialization*.

This diagram shows:
  - the inner-loop adaptation on one task (N=1 view),
  - the outer-loop sampling across N envs,
  - the gradient path that updates only the *initialization* (FOMAML),
  - the relationship to the Tier 0 SSL init (the encoder starts from the
    SSL-pretrained weights, not random).

Output: figures/tier1_sionna/tier1_maml_architecture.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Colors
C_ENC     = "#1f77b4"   # blue
C_HEAD    = "#2ca02c"   # green
C_SSL     = "#9467bd"   # purple
C_TASK    = "#17becf"   # teal
C_INPUT   = "#7f7f7f"   # gray
C_INIT    = "#bcbd22"   # olive
C_LOSS    = "#ff7f0e"   # orange
C_ADAPT   = "#d62728"   # red


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
          label_offset=(0, 0.10), label_color=None, mutation_scale=14):
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="->", mutation_scale=mutation_scale,
        color=color, linewidth=lw, linestyle=style,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arr)
    if label:
        mx = (x0 + x1) / 2 + label_offset[0]
        my = (y0 + y1) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=8,
                color=label_color or color,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))


def main(out_path: str = "figures/tier1_sionna/tier1_maml_architecture.png"):
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("Tier 1 — FOMAML (first-order MAML) over environments",
                 fontsize=15, fontweight="bold", pad=14)

    # ============== Tier 0 SSL init (top) ==============
    box(ax, 7.5, 11.0, 4.5, 0.55, "Tier 0 — SSL pretraining  (mask cells, reconstruct from context)",
        C_SSL, fc="#ebe1f2", fontsize=10, fontweight="bold")
    box(ax, 4.0, 10.2, 2.0, 0.55, "masked X\n(B, T, M, 2)", C_INPUT, fc="#eee")
    box(ax, 7.5, 10.2, 2.4, 0.55, "ASTAEncoder (shared)", C_ENC, fontsize=10, fontweight="bold")
    box(ax, 11.0, 10.2, 2.0, 0.55, "SSLHead + skip\n(pred masked cells)", C_SSL, fontsize=9, fontweight="bold")
    arrow(ax, 5.0, 10.2, 6.3, 10.2, "black", "x_masked")
    arrow(ax, 8.7, 10.2, 10.0, 10.2, C_ENC, "h")
    arrow(ax, 12.0, 10.2, 12.0, 9.5, C_SSL, "x̂")
    box(ax, 12.0, 9.2, 1.8, 0.5, "MSE on masked cells", C_LOSS, fc="#fde9d9", fontsize=9)

    # ============== Shared ASTA encoder box (the parameter being meta-learned) ==============
    box(ax, 8.0, 8.0, 5.0, 0.7,
        "ASTAEncoder  θ_init     ← this is what FOMAML meta-learns",
        C_ENC, fc="#dbe9f5", fontsize=11, fontweight="bold")
    # SSL -> theta_init arrow
    arrow(ax, 7.5, 9.85, 8.0, 8.35, C_SSL, "init weights from", style="--", lw=1.4,
          label_offset=(0, 0.30), label_color=C_SSL, mutation_scale=12)

    # ============== MAML outer loop row (N envs sampled per episode) ==============
    # Sample N envs
    box(ax, 1.5, 6.7, 2.4, 0.55,
        f"sample N={3} envs\n(no replacement)", C_INPUT, fc="#eee", fontsize=9)
    # 3 env boxes
    for i, x in enumerate([4.5, 6.0, 7.5]):
        box(ax, x, 6.7, 1.1, 0.55, f"env {i}", C_TASK, fc="#d6f0ee", fontsize=9)
    arrow(ax, 2.7, 6.7, 3.4, 6.7, "black", "")

    # ============== One task expanded (env 0) — full inner loop ==============
    # support set
    box(ax, 4.5, 5.6, 1.4, 0.5, "support  (K shots)", C_INPUT, fc="#eee", fontsize=9)
    box(ax, 4.5, 5.0, 1.4, 0.5, "query  (Q shots)", C_INPUT, fc="#eee", fontsize=9)
    arrow(ax, 4.5, 6.45, 4.5, 5.85, "black", "split", label_offset=(0.7, 0.0))

    # Inner adaptation block (functional_call, fast weights)
    box(ax, 7.0, 5.6, 2.4, 0.55, "ASTAEncoder  (fast copy)\nθ → θ − α∇L_sup", C_ENC,
        fc="#dbe9f5", fontsize=9, fontweight="bold")
    box(ax, 7.0, 5.0, 2.4, 0.55, "ASTAEncoder  (fast copy)\nθ → θ − α∇L_sup  (step 2)", C_ENC,
        fc="#dbe9f5", fontsize=9)
    # location head
    box(ax, 10.0, 5.6, 2.2, 0.55, "LocationHead  (fast)", C_HEAD, fc="#dfeedd",
        fontsize=9, fontweight="bold")
    box(ax, 10.0, 5.0, 2.2, 0.55, "LocationHead  (fast)", C_HEAD, fc="#dfeedd", fontsize=9)

    # Inner-loss
    box(ax, 12.4, 5.6, 2.0, 0.5, "L_sup  MSE(sup pred, sup y)", C_LOSS, fc="#fde9d9", fontsize=8)
    arrow(ax, 4.5, 5.35, 5.8, 5.6, C_ENC, "sup_x")
    arrow(ax, 5.8, 5.6, 7.0, 5.6, C_ENC)
    arrow(ax, 8.2, 5.6, 8.9, 5.6, C_HEAD)
    arrow(ax, 11.1, 5.6, 11.4, 5.6, C_LOSS, "pred")
    arrow(ax, 5.5, 5.0, 5.8, 5.0, C_ENC, "sup_x")
    arrow(ax, 5.8, 5.0, 7.0, 5.0, C_ENC)
    arrow(ax, 8.2, 5.0, 8.9, 5.0, C_HEAD)
    # dashed grad feedback for inner loop
    arrow(ax, 12.4, 5.35, 8.0, 4.7, C_ADAPT, "−α·∇L_sup", style="--", lw=1.2,
          label_offset=(0, 0.30), label_color=C_ADAPT, mutation_scale=12)

    # Query evaluation (adapted model on query)
    box(ax, 10.0, 4.0, 2.2, 0.55, "ASTA + LocHead\n(fast, post-adapt)", C_ENC,
        fc="#dbe9f5", fontsize=9, fontweight="bold")
    box(ax, 12.4, 4.0, 2.0, 0.5, "L_q  MSE(q pred, q y)", C_LOSS, fc="#fde9d9", fontsize=8)
    arrow(ax, 4.5, 4.75, 8.9, 4.0, C_ENC, "q_x")
    arrow(ax, 11.1, 4.0, 11.4, 4.0, C_LOSS, "pred")

    # ============== Repeat for env 1, env 2 (stacked) — show that all 3 share θ_init ==============
    # We don't draw full inner loops for envs 1,2 to keep the figure readable.
    # Instead, show a "stack of tasks" hint with env1/env2 boxes and arrows
    # going to a common θ_init update.
    for i, (x, y) in enumerate([(5.0, 3.0), (8.0, 3.0), (11.0, 3.0)]):
        box(ax, x, y, 2.0, 0.5, f"env {i}: L_q{i+1}", C_LOSS, fc="#fde9d9", fontsize=9)
    box(ax, 8.0, 1.9, 4.5, 0.55,
        "Σ L_q  /  N  →  ∂/∂θ  →  θ ← θ − β·∇   (FOMAML outer step)",
        C_INIT, fc="#ecebe0", fontsize=10, fontweight="bold")
    # Pull L_q from each env into the outer sum
    for x, y in [(5.0, 3.0), (8.0, 3.0), (11.0, 3.0)]:
        arrow(ax, x, 2.75, 8.0, 2.18, C_LOSS, style="-", lw=1.0, mutation_scale=10)

    # Update arrow back to θ_init
    arrow(ax, 8.0, 1.6, 8.0, 7.65, C_INIT, "θ_init updated", style="--", lw=1.6,
          label_offset=(0.6, 0), label_color=C_INIT, mutation_scale=14)

    # ============== Sidebar ==============
    note = (
        "MAML is *learning an initialization* θ_init, not a model.\n"
        "\n"
        "Inner loop (per task):\n"
        "  • functional_call builds a fast-weight copy of θ_init\n"
        "  • K gradient steps on support set:  θ ← θ − α·∇L_sup\n"
        "  • L_sup drives the fast weights; not used to update θ_init\n"
        "\n"
        "Outer loop (per episode):\n"
        "  • sample N envs, sum their query losses\n"
        "  • ∂L_q/∂θ is taken W.R.T. THE INITIAL θ (FOMAML = 1st order)\n"
        "  • θ_init ← θ_init − β·∇ΣL_q / N\n"
        "\n"
        "What FOMAML gives up vs full MAML:\n"
        "  • discards the second-order (Hessian-through-inner-loop) terms\n"
        "  • much cheaper, more stable, often just as good in practice\n"
        "\n"
        "Key config (ultraclear/config.py):\n"
        "  n_way=3, k_shot, q_query, maml_inner_lr, maml_inner_steps,\n"
        "  maml_outer_lr, maml_episodes"
    )
    ax.text(0.3, 4.7, note, fontsize=8.5, va="top", ha="left", family="monospace",
            bbox=dict(facecolor="white", edgecolor="#aaa", boxstyle="round,pad=0.6"))

    # ============== Legend ==============
    lx, ly = 12.0, 11.0
    ax.text(lx, ly, "Legend:", fontsize=9, fontweight="bold")
    ax.plot([lx + 0.8, lx + 1.6], [ly, ly], color="black", lw=1.6)
    ax.text(lx + 1.7, ly, "forward (data)", fontsize=8, va="center")
    ax.plot([lx + 0.8, lx + 1.6], [ly - 0.25, ly - 0.25],
            color=C_ADAPT, lw=1.4, linestyle="--")
    ax.text(lx + 1.7, ly - 0.25, "inner-loop grad", fontsize=8, va="center")
    ax.plot([lx + 0.8, lx + 1.6], [ly - 0.50, ly - 0.50],
            color=C_INIT, lw=1.4, linestyle="--")
    ax.text(lx + 1.7, ly - 0.50, "outer-loop grad → θ_init", fontsize=8, va="center")
    ax.plot([lx + 0.8, lx + 1.6], [ly - 0.75, ly - 0.75],
            color=C_SSL, lw=1.4, linestyle="--")
    ax.text(lx + 1.7, ly - 0.75, "SSL pretraining → init", fontsize=8, va="center")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
