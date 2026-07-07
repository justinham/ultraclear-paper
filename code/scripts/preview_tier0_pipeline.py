"""Preview the Tier 0 pipeline figure as a PNG.

Left-to-right pipeline layout, no crossing arrows:
  [raw -> mask -> masked]  ->  [ASTA encoder]  ->  [SSL head]  ->  [loss]
                                       |                               ^
                                       v                               |
                              [pretrained encoder]  (reused downstream)

The "true X" path is the only long arrow (data flow to loss, dashed over
the top, well-separated from the forward flow).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch


# Colors matching architecture.tex
COLOR_RAW = "#cfe2f3"   # blue!22
COLOR_ASTA_INNER = "#deebf7"   # blue!12
COLOR_GCN = "#b2d8d8"   # teal!18
COLOR_MHSA = "#fce5cd"   # orange!18
COLOR_HEAD = "#deebf7"
COLOR_LOSS = "#f4cccc"   # red!14
COLOR_OUT = "#b2d8d8"   # teal!22
COLOR_VISIBLE = "#deebf7"
COLOR_AOA_VISIBLE = "#fce5cd"
COLOR_MASKED = "#1a1a1a"


def draw_window(ax, x0, y0, mask_pattern, fill=COLOR_VISIBLE, cell=2.4):
    """Draw an 8x4 grid at (x0, y0). mask_pattern is (8, 4) bool."""
    T, M = mask_pattern.shape
    for t in range(T):
        for m in range(M):
            x = x0 + t * cell
            y = y0 - m * cell
            color = COLOR_MASKED if mask_pattern[t, m] else fill
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor=color,
                                    edgecolor="#888", linewidth=0.3))


def draw_box(ax, x, y, w, h, text, color=COLOR_ASTA_INNER, fontsize=9,
              text_color="black", weight="normal"):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.02,rounding_size=1.2",
                          facecolor=color, edgecolor="#666", linewidth=0.8)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight=weight)


def draw_arrow(ax, p0, p1, color="#333", lw=1.4, style="-|>", text=None,
                text_offset=(0, 1.5), text_color=None, text_fontsize=8):
    arr = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                           color=color, linewidth=lw, shrinkA=0, shrinkB=0)
    ax.add_patch(arr)
    if text:
        c = text_color if text_color else color
        ax.text((p0[0] + p1[0])/2 + text_offset[0],
                (p0[1] + p1[1])/2 + text_offset[1],
                text, ha="center", va="center", fontsize=text_fontsize,
                color=c, fontstyle="italic")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(-2, 50)
    ax.set_aspect("equal")
    ax.axis("off")

    cell = 2.4
    T, M = 8, 4
    # 25% mask
    mask_pattern = np.zeros((T, M), dtype=bool)
    for t, m in [(0,1), (0,3), (1,2), (2,2), (3,0), (4,1),
                  (5,2), (6,2), (7,0), (7,3)]:
        mask_pattern[t, m] = True
    n_masked = int(mask_pattern.sum())
    print(f"mask: {n_masked}/{T*M} ({100*n_masked/(T*M):.0f}%)")

    # ============================================================
    # COLUMN 1: data prep (stacked vertically)
    # x range: 0-25
    # ============================================================
    cx_data = 12.5  # center x for data column

    # 1a: raw window (ToA on top, AoA below as a second row)
    # ToA
    x0, y0 = 1.5, 32
    for t in range(T):
        for m in range(M):
            x = x0 + t * cell
            y = y0 - m * cell
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor=COLOR_VISIBLE,
                                    edgecolor="#888", linewidth=0.3))
    # AoA below
    y0_aoa = y0 - M * cell - 1
    for t in range(T):
        for m in range(M):
            x = x0 + t * cell
            y = y0_aoa - m * cell
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor=COLOR_AOA_VISIBLE,
                                    edgecolor="#888", linewidth=0.3))
    ax.text(x0 + T * cell / 2, y0 + 1.5, r"raw window $\mathbf{X}$",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
    # ToA / AoA labels on the right
    ax.text(x0 + T * cell + 1.5, y0 - 1.2, "ch 0\nToA (m)",
            ha="left", va="center", fontsize=7, color="#1a4d8c")
    ax.text(x0 + T * cell + 1.5, y0_aoa - 1.2, "ch 1\nAoA (°)",
            ha="left", va="center", fontsize=7, color="#a85b00")
    # shape annotation
    ax.text(x0 + T * cell / 2, y0_aoa - M * cell - 1.5,
            r"$(T{=}32,\, M{=}4,\, F{=}2)$",
            ha="center", va="top", fontsize=7, color="#555", fontstyle="italic")

    # 1b: random mask
    x0_m = 1.5
    y0_m = 16
    for t in range(T):
        for m in range(M):
            x = x0_m + t * cell
            y = y0_m - m * cell
            color = COLOR_MASKED if mask_pattern[t, m] else "#f5f5f5"
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor=color,
                                    edgecolor="#888", linewidth=0.3))
    ax.text(x0_m + T * cell / 2, y0_m + 1.5, r"random mask $\mathbf{M}$",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(x0_m + T * cell + 1.5, y0_m - 2,
            r"$(T, M)$ bool, $p{=}0.25$",
            ha="left", va="center", fontsize=7, color="#555", fontstyle="italic")
    ax.text(x0_m + T * cell + 1.5, y0_m - 5,
            "joint across $F$",
            ha="left", va="center", fontsize=7, color="#a52a2a", fontstyle="italic")

    # 1c: masked input
    x0_mi = 1.5
    y0_mi = 0
    for t in range(T):
        for m in range(M):
            x = x0_mi + t * cell
            y = y0_mi - m * cell
            color = COLOR_MASKED if mask_pattern[t, m] else COLOR_VISIBLE
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor=color,
                                    edgecolor="#888", linewidth=0.3))
    y0_mi_aoa = y0_mi - M * cell - 1
    for t in range(T):
        for m in range(M):
            x = x0_mi + t * cell
            y = y0_mi_aoa - m * cell
            color = COLOR_MASKED if mask_pattern[t, m] else COLOR_AOA_VISIBLE
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor=color,
                                    edgecolor="#888", linewidth=0.3))
    ax.text(x0_mi + T * cell / 2, y0_mi - M * cell - 3,
            r"masked input $\tilde{\mathbf{X}}{=}\mathbf{X}\!\odot\!(1{-}\mathbf{M})$",
            ha="center", va="top", fontsize=10, fontweight="bold")

    # arrows within column 1
    draw_arrow(ax, (cx_data, 28), (cx_data, 19), lw=1.4)         # raw -> mask
    ax.text(cx_data + 1.5, 23.5, "sample", fontsize=8, color="#555", fontstyle="italic")
    draw_arrow(ax, (cx_data, 12), (cx_data, 3.5), lw=1.4)         # mask -> masked
    ax.text(cx_data + 1.5, 7.5, "apply", fontsize=8, color="#555", fontstyle="italic")

    # ============================================================
    # COLUMN 2: ASTA encoder
    # ============================================================
    cx_enc = 40
    # title
    ax.text(cx_enc, 38, r"ASTA encoder $f_\theta$",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(cx_enc, 36.5, r"$(T, M, F) \to (T, M, d{=}64)$",
            ha="center", va="top", fontsize=8, color="#555", fontstyle="italic")
    # 3 blocks stacked
    draw_box(ax, cx_enc, 32, 18, 3.5,
              r"Causal TCN" + "\n" + r"per-channel, $k{=}3$",
              color=COLOR_ASTA_INNER, fontsize=8)
    draw_box(ax, cx_enc, 26, 18, 3.5,
              r"Dynamic GCN" + "\n" + r"$A^{(c)}$, per-channel",
              color=COLOR_GCN, fontsize=8)
    draw_box(ax, cx_enc, 20, 18, 3.5,
              r"Multi-Head Self-Attn" + "\n" + r"$h{=}4$ heads, $d{=}64$",
              color=COLOR_MHSA, fontsize=8)
    # internal arrows
    draw_arrow(ax, (cx_enc, 30.2), (cx_enc, 27.8), lw=1.4)
    draw_arrow(ax, (cx_enc, 24.2), (cx_enc, 21.8), lw=1.4)
    # residual: bracket on the right
    draw_arrow(ax, (cx_enc + 9.2, 32), (cx_enc + 9.2, 20),
                color="#1a4d8c", lw=0.9, style="-|>")
    ax.text(cx_enc + 10.5, 26, "+ residual", fontsize=7, color="#1a4d8c",
            ha="left", va="center", fontstyle="italic")

    # ============================================================
    # COLUMN 3: SSL head
    # ============================================================
    cx_head = 62
    ax.text(cx_head, 38, r"SSL head $g$",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(cx_head, 36.5, r"2-layer MLP, $d \to d \to F$",
            ha="center", va="top", fontsize=8, color="#555", fontstyle="italic")
    draw_box(ax, cx_head, 28, 18, 3.5,
              r"Linear$(d{\to}d) \to$ GELU",
              color=COLOR_HEAD, fontsize=8)
    draw_box(ax, cx_head, 22, 18, 3.5,
              r"Linear$(d{\to}F)$",
              color=COLOR_HEAD, fontsize=8)
    draw_arrow(ax, (cx_head, 26.2), (cx_head, 23.8), lw=1.4)
    # output annotation
    ax.text(cx_head + 10, 22, r"$\hat{\mathbf{X}}$" + "\n" + r"(ToA + AoA)",
            fontsize=8, color="#a85b00", ha="left", va="center",
            fontstyle="italic")

    # ============================================================
    # COLUMN 4: loss
    # ============================================================
    cx_loss = 85
    ax.text(cx_loss, 38, "loss",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
    draw_box(ax, cx_loss, 32, 20, 5.5,
              r"$\mathcal{L} = \frac{\sum \mathbf{M} \odot (\hat{\mathbf{X}} - \mathbf{X})^2}{\sum \mathbf{M}}$",
              color=COLOR_LOSS, fontsize=10, weight="bold")
    ax.text(cx_loss, 27,
              r"per-channel weights" + "\n" + r"$w_{\mathrm{ToA}}{=}3$, $w_{\mathrm{AoA}}{=}1$",
              ha="center", va="center", fontsize=7, color="#a52a2a",
              fontstyle="italic")
    draw_box(ax, cx_loss, 21, 18, 3.5,
              r"Adam, $\eta{=}10^{-3}$" + "\n" + r"$30{-}100$ epochs",
              color=COLOR_LOSS, fontsize=8)
    draw_arrow(ax, (cx_loss, 29.2), (cx_loss, 22.8), lw=1.4)

    # ============================================================
    # FORWARD ARROWS (column-to-column)
    # ============================================================
    # column 1 -> column 2 (masked -> encoder)
    # 2-segment path: right from masked input (y=mid), then up to TCN (y=32)
    # start at right of masked input, mid-AoA-row
    start_y = (y0_mi + y0_mi_aoa) / 2  # midway between ToA and AoA midpoints
    mid_x = (x0_mi + T * cell + cx_enc - 9) / 2
    draw_arrow(ax, (x0_mi + T * cell + 1, start_y), (mid_x, start_y),
                lw=1.4)
    draw_arrow(ax, (mid_x, start_y), (mid_x, 32),
                lw=1.4)
    draw_arrow(ax, (mid_x, 32), (cx_enc - 9, 32),
                lw=1.4)
    ax.text(mid_x, start_y - 2, "encode", fontsize=8, color="#555",
            ha="center", fontstyle="italic")

    # column 2 -> column 3 (encoder MHSA -> head Linear1)
    draw_arrow(ax, (cx_enc + 9, 20), (cx_head - 9, 28),
                lw=1.4, text=r"$\mathbf{H}$",
                text_offset=(0, 1.5), text_color="#0d6b6b", text_fontsize=10)

    # column 3 -> column 4 (head output -> loss)
    draw_arrow(ax, (cx_head + 9, 22), (cx_loss - 10, 32),
                lw=1.4, text=r"$\hat{\mathbf{X}}$",
                text_offset=(0, 1.5), text_color="#a85b00", text_fontsize=9)

    # column 4 -> output (Adam -> pretrained encoder)
    # ============================================================
    # BELOW column 2: pretrained encoder (reused downstream)
    # ============================================================
    cy_out = 6
    out = FancyBboxPatch((cx_enc - 12, cy_out - 3), 24, 6,
                          boxstyle="round,pad=0.02,rounding_size=1.2",
                          facecolor=COLOR_OUT, edgecolor="#666", linewidth=1.0)
    ax.add_patch(out)
    ax.text(cx_enc, cy_out,
            r"pretrained encoder $f_{\theta^*}$" + "\n" + r"(reused by Tier 1 / 3)",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="#0d4f4f")

    # update arrow from Adam (cx_loss, 21 - h/2 = 19.25) down/left to encoder output
    # the encoder MHSA is at (cx_enc, 20) with bottom at 18.25
    # route: from Adam left, down, then into encoder output from above
    arr = FancyArrowPatch((cx_loss - 9, 21 - 1.75), (cx_enc + 9, 6),
                           arrowstyle="-|>", mutation_scale=14,
                           color="#333", linewidth=1.4,
                           connectionstyle="arc3,rad=-0.2")
    ax.add_patch(arr)
    ax.text(cx_loss - 5, 13, "update", ha="center", va="center", fontsize=9,
            color="#333", fontstyle="italic")
    # also show "the encoder f_theta updates" via a direct loop arrow
    # from MHSA back down to the pretrained encoder box
    draw_arrow(ax, (cx_enc, 18.25), (cx_enc, 9),
                color="#0d6b6b", lw=1.0, style="-|>",
                text="parameters", text_offset=(2, -1), text_color="#0d6b6b",
                text_fontsize=8)

    # ============================================================
    # TRUE-X dashed path: from raw window, over the top, down to loss
    # ============================================================
    # start at top of raw window (x0, y0+1), go up to y=46, right to loss, down
    true_x_start = (x0 + T * cell / 2, y0 + 0.5)
    # route: up to y=45, right to x=cx_loss, down to top of loss
    draw_arrow(ax, true_x_start, (true_x_start[0], 45),
                color="#1a4d8c", lw=0.9, style="-|>", text=None)
    draw_arrow(ax, (true_x_start[0], 45), (cx_loss, 45),
                color="#1a4d8c", lw=0.9, style="-|>")
    draw_arrow(ax, (cx_loss, 45), (cx_loss, 34.75),
                color="#1a4d8c", lw=0.9, style="-|>")
    ax.text(50, 46.5, r"true $\mathbf{X}$ (from data)",
            ha="center", va="bottom", fontsize=8, color="#1a4d8c",
            fontstyle="italic")

    # Title
    fig.suptitle(r"Tier 0 — Self-supervised masked-prediction pretraining",
                  fontsize=12, fontweight="bold", y=0.98)

    out_path = Path(args.out) if args.out else Path(__file__).parent.parent / "figures" / "tier0_pipeline_preview.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
