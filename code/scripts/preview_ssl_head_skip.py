"""Preview the SSL head WITH the hard skip-connection.

The skip-connection fix is the key new piece: at visible cells
(where mask is False), the head output equals the raw input x
(skip-path). At masked cells (where mask is True), the head output
equals MLP(encoder(x_masked)) (inference-path). The result is that
the SSL pretraining is exactly equivalent to a denoising task on
masked cells, with a hard identity on visible cells.

This figure is the "v2" SSL head architecture, side-by-side with
the "v1" (no-skip) head for contrast.

Layout (left-to-right, two stacked panels):
  TOP panel (v1, NO skip): raw X  --+--> masked X  -->  encoder  -->  MLP  -->  X_hat
                                    |                                              |
                                    +---------------------------- (skip) ---------+
                                                              (over the top)
  BOTTOM panel (v2, WITH skip): raw X --+--> masked X  -->  encoder  -->  MLP  -->  X_hat
                                       |                                              |
                                       |                                  WHERE M: X_hat
                                       |                                  ELSE:    X
                                       +--->------------------------------------+

The key visual: in v2, the skip goes DIRECTLY into a "WHERE M:
X_hat, ELSE: X" merge node. In v1, the skip is shown as a "would-
be" connection that is NOT made.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, FancyArrow

# Colors (match architecture.tex palette)
COLOR_RAW = "#cfe2f3"            # blue!22
COLOR_MASKED = "#1a1a1a"         # solid black
COLOR_VISIBLE = "#deebf7"        # blue!12
COLOR_AOA_VISIBLE = "#fce5cd"    # orange!18
COLOR_ENCODER = "#cfe2f3"        # blue!22
COLOR_HEAD = "#deebf7"           # blue!12
COLOR_SKIP = "#a52a2a"           # red (the skip is the new piece)
COLOR_MERGE = "#fff2cc"          # yellow!15 (the WHERE node)
COLOR_OUT = "#b2d8d8"            # teal!22


def draw_window(ax, x0, y0, mask_pattern, fill=COLOR_VISIBLE,
                 aoa_fill=COLOR_AOA_VISIBLE, cell=1.8, show_aoa=True):
    """Draw a (T, M) grid at (x0, y0), with ToA on top and AoA below."""
    T, M = mask_pattern.shape
    for t in range(T):
        for m in range(M):
            x = x0 + t * cell
            y = y0 - m * cell
            color = COLOR_MASKED if mask_pattern[t, m] else fill
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor=color,
                                    edgecolor="#888", linewidth=0.3))
    if show_aoa:
        y0_aoa = y0 - M * cell - 0.6
        for t in range(T):
            for m in range(M):
                x = x0 + t * cell
                y = y0_aoa - m * cell
                color = COLOR_MASKED if mask_pattern[t, m] else aoa_fill
                ax.add_patch(Rectangle((x, y), cell, cell, facecolor=color,
                                        edgecolor="#888", linewidth=0.3))
        return y0_aoa - M * cell
    return y0 - M * cell


def draw_box(ax, x, y, w, h, text, color=COLOR_HEAD, fontsize=8,
              text_color="black", weight="normal"):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.6",
                          facecolor=color, edgecolor="#555", linewidth=0.8)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight=weight)


def draw_arrow(ax, p0, p1, color="#333", lw=1.4, style="-|>", text=None,
                text_offset=(0, 0.5), text_color=None, text_fontsize=7,
                linestyle="-"):
    arr = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=12,
                           color=color, linewidth=lw, shrinkA=0, shrinkB=0,
                           linestyle=linestyle)
    ax.add_patch(arr)
    if text:
        c = text_color if text_color else color
        ax.text((p0[0] + p1[0])/2 + text_offset[0],
                (p0[1] + p1[1])/2 + text_offset[1],
                text, ha="center", va="center", fontsize=text_fontsize,
                color=c, fontstyle="italic")


def make_mask():
    """A (8, 4) mask pattern with ~25% cells black."""
    mask = np.zeros((8, 4), dtype=bool)
    for t, m in [(0,1), (0,3), (1,2), (2,2), (3,0), (4,1),
                  (5,2), (6,2), (7,0), (7,3)]:
        mask[t, m] = True
    return mask


def draw_panel(ax, y_top, y_bot, panel_label, show_skip, mask_pattern):
    """Draw one panel (v1 or v2) of the SSL head diagram.
    y_top, y_bot: vertical extent of the panel in figure coords.
    """
    T, M = mask_pattern.shape
    cell = 1.8

    # ============================================================
    # Column 1: raw window + masked window (vertical stack)
    # ============================================================
    x0 = 2
    y_raw = y_top - 2
    y_masked = y_bot + 2

    # 1a: raw window (ToA + AoA)
    y_raw_bottom = draw_window(ax, x0, y_raw, np.zeros_like(mask_pattern), cell=cell)
    ax.text(x0 + T * cell / 2, y_raw + 1.2, r"raw $\mathbf{X}$",
            ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.text(x0 + T * cell + 0.6, (y_raw + y_raw_bottom)/2, "(T,M,F)\ninput",
            ha="left", va="center", fontsize=6, color="#555", fontstyle="italic")

    # 1b: masked window
    y_masked_bottom = draw_window(ax, x0, y_masked, mask_pattern, cell=cell)
    ax.text(x0 + T * cell / 2, y_masked - 1.2, r"masked $\tilde{\mathbf{X}}$",
            ha="center", va="top", fontsize=8, fontweight="bold")
    ax.text(x0 + T * cell + 0.6, (y_masked + y_masked_bottom)/2,
             "M=1 cells\nzeroed",
             ha="left", va="center", fontsize=6, color="#a52a2a",
             fontstyle="italic")

    # Arrow: raw -> masked (apply mask)
    raw_center = (x0 + T * cell / 2, (y_raw + y_raw_bottom) / 2)
    masked_center = (x0 + T * cell / 2, (y_masked + y_masked_bottom) / 2)
    draw_arrow(ax, (raw_center[0], raw_center[1] - 1),
                (masked_center[0], masked_center[1] + 1),
                lw=1.2, text="apply mask $\\mathbf{M}$",
                text_offset=(0, 0.3), text_fontsize=7, text_color="#555")

    # ============================================================
    # Column 2: ASTA encoder (single block, not the full stack)
    # ============================================================
    cx_enc = 18
    cy_enc = (y_top + y_bot) / 2
    draw_box(ax, cx_enc, cy_enc, 6.5, 4.5,
              r"ASTA encoder" + "\n" + r"$f_\theta$",
              color=COLOR_ENCODER, fontsize=8, weight="bold")
    ax.text(cx_enc, cy_enc - 3.0, r"$(T, M, F) \to \mathbf{H}$",
            ha="center", va="top", fontsize=6, color="#555", fontstyle="italic")

    # Arrow: masked -> encoder
    masked_right = (x0 + T * cell + 0.6, masked_center[1])
    enc_left = (cx_enc - 3.25, cy_enc)
    draw_arrow(ax, masked_right, enc_left, lw=1.2,
                text="encode", text_offset=(0, 0.3), text_color="#555")

    # ============================================================
    # Column 3: MLP head
    # ============================================================
    cx_head = 28
    cy_head = cy_enc
    draw_box(ax, cx_head, cy_head, 6, 4,
              r"2-layer MLP" + "\n" + r"$g$",
              color=COLOR_HEAD, fontsize=8, weight="bold")
    ax.text(cx_head, cy_head - 2.7, r"$\mathbf{H} \to F$",
            ha="center", va="top", fontsize=6, color="#555", fontstyle="italic")

    # Arrow: encoder -> head
    draw_arrow(ax, (cx_enc + 3.25, cy_enc), (cx_head - 3, cy_head),
                lw=1.2, text=r"$\mathbf{H}$", text_offset=(0, 0.3),
                text_color="#0d6b6b")

    # Output of MLP (the predicted X_hat for ALL cells)
    head_right = (cx_head + 3, cy_head)
    ax.text(head_right[0] + 0.5, cy_head + 1.0,
             r"$\hat{\mathbf{X}}_{\mathrm{MLP}}$",
             ha="left", va="center", fontsize=7, color="#0d6b6b",
             fontstyle="italic")
    ax.text(head_right[0] + 0.5, cy_head - 0.7,
             r"(inferred values)",
             ha="left", va="center", fontsize=6, color="#555",
             fontstyle="italic")

    # ============================================================
    # Column 4: WHERE node (merge) -- this is the new piece
    # ============================================================
    cx_merge = 40
    cy_merge = cy_enc
    if show_skip:
        # Active merge node
        draw_box(ax, cx_merge, cy_merge, 6, 5,
                  r"$\hat{\mathbf{X}}$" + "\n" + r"$=$ WHERE($\mathbf{M}$," + "\n" +
                  r"$\hat{\mathbf{X}}_{\mathrm{MLP}},\,\mathbf{X}$)",
                  color=COLOR_MERGE, fontsize=7, weight="bold",
                  text_color="#5c4d00")
        ax.text(cx_merge, cy_merge - 3.0,
                 r"$\mathbf{M}{=}1$: $\hat{\mathbf{X}}_{\mathrm{MLP}}$" + "\n" +
                 r"$\mathbf{M}{=}0$: $\mathbf{X}$ (passthrough)",
                 ha="center", va="top", fontsize=6, color="#5c4d00",
                 fontstyle="italic")
    else:
        # v1: just a passthrough box, no merge
        draw_box(ax, cx_merge, cy_merge, 5, 3.5,
                  r"$\hat{\mathbf{X}}$",
                  color="#f5f5f5", fontsize=8, weight="bold")
        ax.text(cx_merge, cy_merge - 2.5, r"$= \hat{\mathbf{X}}_{\mathrm{MLP}}$",
                 ha="center", va="top", fontsize=6, color="#555",
                 fontstyle="italic")

    # Arrow: head -> merge (X_hat_MLP path)
    draw_arrow(ax, head_right, (cx_merge - 3, cy_merge),
                lw=1.2, color="#0d6b6b")

    # ============================================================
    # SKIP path: from raw X (top-left) to merge node (right)
    # ============================================================
    # The skip starts at the right edge of the raw X window, goes
    # up over the encoder/head, and comes down into the merge.
    raw_right = (x0 + T * cell + 0.6, (y_raw + y_raw_bottom) / 2)
    merge_left = (cx_merge - 3, cy_merge)
    if show_skip:
        # Active: solid red line
        skip_top = y_top + 1.0
        # route: from raw_right, up to skip_top, right to above merge, down
        draw_arrow(ax, raw_right, (raw_right[0], skip_top),
                    color=COLOR_SKIP, lw=1.6, style="-|>")
        draw_arrow(ax, (raw_right[0], skip_top), (merge_left[0] - 1, skip_top),
                    color=COLOR_SKIP, lw=1.6, style="-|>")
        draw_arrow(ax, (merge_left[0] - 1, skip_top), merge_left,
                    color=COLOR_SKIP, lw=1.6, style="-|>")
        ax.text((raw_right[0] + merge_left[0]) / 2, skip_top + 0.5,
                 r"skip: raw $\mathbf{X}$ at visible cells",
                 ha="center", va="bottom", fontsize=7, color=COLOR_SKIP,
                 fontweight="bold")
    else:
        # v1: gray dashed line, clearly inactive
        skip_top = y_top + 1.0
        draw_arrow(ax, raw_right, (raw_right[0], skip_top),
                    color="#999", lw=0.9, style="-|>", linestyle="--")
        draw_arrow(ax, (raw_right[0], skip_top), (merge_left[0] - 1, skip_top),
                    color="#999", lw=0.9, style="-|>", linestyle="--")
        # In v1, the skip ENDS at the merge's top edge (not entering).
        # We draw a small "X" or annotation to show it's not connected.
        ax.text((raw_right[0] + merge_left[0]) / 2, skip_top + 0.4,
                 r"skip NOT wired in v1 (encoder output used everywhere)",
                 ha="center", va="bottom", fontsize=6, color="#999",
                 fontstyle="italic")
        # small "X" marker at merge input
        ax.plot([merge_left[0] - 0.4, merge_left[0] - 0.05],
                 [skip_top - 0.3, skip_top + 0.3],
                 color="#999", lw=1.2)
        ax.plot([merge_left[0] - 0.4, merge_left[0] - 0.05],
                 [skip_top + 0.3, skip_top - 0.3],
                 color="#999", lw=1.2)

    # ============================================================
    # Panel label (left side)
    # ============================================================
    if show_skip:
        ax.text(0.2, cy_enc, panel_label, ha="left", va="center",
                 fontsize=11, fontweight="bold", color="#0d6b6b",
                 rotation=90)
    else:
        ax.text(0.2, cy_enc, panel_label, ha="left", va="center",
                 fontsize=11, fontweight="bold", color="#888",
                 rotation=90)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    args = p.parse_args()

    fig = plt.figure(figsize=(13, 7.5))
    ax = fig.add_subplot(111)
    ax.set_xlim(-1, 50)
    ax.set_ylim(0, 30)
    ax.set_aspect("equal")
    ax.axis("off")

    mask_pattern = make_mask()

    # Title
    fig.suptitle("SSL head: with and without the hard skip-connection",
                  fontsize=13, fontweight="bold", y=0.97)
    fig.text(0.5, 0.93,
              "Red path = the new piece. Solid in v2, dashed in v1.",
              ha="center", fontsize=9, color="#555", style="italic")

    # Two stacked panels
    # v1 (no skip) on TOP, v2 (with skip) on BOTTOM
    y_top_v1 = 27
    y_bot_v1 = 16
    draw_panel(ax, y_top_v1, y_bot_v1, "v1: NO skip", show_skip=False, mask_pattern=mask_pattern)

    y_top_v2 = 12
    y_bot_v2 = 1
    draw_panel(ax, y_top_v2, y_bot_v2, "v2: WITH skip", show_skip=True, mask_pattern=mask_pattern)

    # Bottom: side-by-side masked/unmasked outcome
    # (mini result boxes showing the effect)
    y_box = 0.2
    box_w = 7
    box_h = 0.7
    # v1 box
    draw_box(ax, 11, y_box, box_w, box_h,
              r"v1:  ToA 1.78m @ visible",
              color="#fce4e4", fontsize=7, text_color="#a52a2a")
    # v2 box
    draw_box(ax, 30, y_box, box_w, box_h,
              r"v2:  ToA 0.00m @ visible",
              color="#e6f4ea", fontsize=7, text_color="#0d6b3b")

    out_path = Path(args.out) if args.out else Path(__file__).resolve().parent.parent.parent / "figures" / "tier0_ssl_head_skip_preview.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
