"""Clean, paper-quality boxes-and-arrows diagram of the SSL head with skip.

Two stacked panels: v1 (no skip) and v2 (with hard skip-connection).
Layout: boxes, arrows, and result badges with explicit vertical clearance.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Colors (match architecture.tex palette)
C_RAW = "#cfe2f3"
C_ENC = "#cfe2f3"
C_HEAD = "#deebf7"
C_MERGE_ON = "#fff2cc"
C_MERGE_OFF = "#f5f5f5"
C_RESULT_BAD = "#f4cccc"
C_RESULT_GOOD = "#d9ead3"
C_SKIP = "#a52a2a"
C_ARROW_DATA = "#1a4d8c"


def box(ax, x, y, w, h, text, fc="#deebf7", ec="#555", fontsize=10,
         weight="normal", text_color="black"):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle="round,pad=0.02,rounding_size=0.15",
                        facecolor=fc, edgecolor=ec, linewidth=1.0)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight=weight)


def arrow(ax, p0, p1, color="#333", lw=1.5, style="-|>", linestyle="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                        color=color, linewidth=lw, shrinkA=0, shrinkB=0,
                        linestyle=linestyle)
    ax.add_patch(a)


def label(ax, x, y, text, color="#555", fontsize=8, style="italic", weight="normal"):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=color, fontstyle=style, fontweight=weight)


def draw_panel(ax, yc, show_skip, panel_label, panel_label_color):
    """Draw one panel at vertical center yc.
    show_skip: if True, draw the solid red skip path into a WHERE merge.
               if False, draw the dashed gray skip path with an X (inactive)."""
    # Pipeline y positions: encoder/head/merge on the same row, raw above, masked below
    y_row = yc
    y_raw = yc + 1.8
    y_mask = yc - 1.8
    # Pipeline x positions
    x_x, x_mask, x_enc, x_head, x_merge = 0.5, 3.5, 7.0, 10.5, 14.5

    # ============ Boxes (pipeline row) ============
    box(ax, x_x, y_row, 1.4, 0.85, r"raw $\mathbf{X}$",
        fc=C_RAW, ec=C_ARROW_DATA, fontsize=10, weight="bold")
    box(ax, x_mask, y_row, 1.6, 0.85, r"masked $\tilde{\mathbf{X}}$",
        fc=C_RAW, ec=C_ARROW_DATA, fontsize=10, weight="bold")
    box(ax, x_enc, y_row, 1.8, 1.1, r"ASTA encoder" + "\n" + r"$f_\theta$",
        fc=C_ENC, ec=C_ARROW_DATA, fontsize=9, weight="bold")
    box(ax, x_head, y_row, 1.7, 1.1, r"MLP $g$" + "\n" + r"(2-layer)",
        fc=C_HEAD, fontsize=9, weight="bold")

    if show_skip:
        # WHERE merge — wider, with formula
        box(ax, x_merge, y_row, 2.6, 1.4,
            "WHERE merge\n"
            r"$\hat{\mathbf{X}} = \mathrm{where}($" + "\n"
            r"$\mathbf{M},\;\hat{\mathbf{X}}_{\mathrm{MLP}},\;\mathbf{X})$",
            fc=C_MERGE_ON, ec="#a08500", fontsize=8, weight="bold",
            text_color="#5c4d00")
    else:
        box(ax, x_merge, y_row, 1.8, 0.85, r"$\hat{\mathbf{X}}$",
            fc=C_MERGE_OFF, ec="#888", fontsize=10, weight="bold")

    # ============ Pipeline arrows (top row, left to right) ============
    arrow(ax, (x_x + 0.7, y_row), (x_mask - 0.8, y_row), color=C_ARROW_DATA)
    arrow(ax, (x_mask + 0.8, y_row), (x_enc - 0.9, y_row), color=C_ARROW_DATA)
    arrow(ax, (x_enc + 0.9, y_row), (x_head - 0.85, y_row), color=C_ARROW_DATA)
    arrow(ax, (x_head + 0.85, y_row), (x_merge - (1.3 if show_skip else 0.9), y_row),
          color=C_ARROW_DATA)

    label(ax, (x_x + x_mask) / 2, y_row + 0.6,
          r"apply $\mathbf{M}$", color=C_ARROW_DATA, fontsize=7)
    label(ax, (x_mask + x_enc) / 2, y_row + 0.6,
          r"encode", color=C_ARROW_DATA, fontsize=7)
    label(ax, (x_enc + x_head) / 2, y_row + 0.7,
          r"$\mathbf{H}$", color="#0d6b6b", fontsize=8, style="normal")
    label(ax, (x_head + x_merge) / 2, y_row + 0.7,
          r"$\hat{\mathbf{X}}_{\mathrm{MLP}}$", color="#0d6b6b",
          fontsize=7, style="normal")

    # ============ Show what each (T, M) cell looks like at the masked X box ============
    # Small inset: a (T=2, M=2) cell grid showing 1 masked cell out of 4
    inset_cx, inset_cy = x_mask, y_row - 1.55
    cell = 0.25
    for t in range(2):
        for m in range(2):
            cx = inset_cx + (t - 0.5) * cell
            cy = inset_cy + (m - 0.5) * cell
            # 1 of 4 is masked
            color = "#1a1a1a" if (t == 0 and m == 0) else "#cfe2f3"
            ax.add_patch(plt.Rectangle((cx - cell/2, cy - cell/2), cell, cell,
                                        facecolor=color, edgecolor="#888", lw=0.3))
    label(ax, inset_cx + 0.8, inset_cy, "(black = M=1,\nblue = M=0)",
          color="#555", fontsize=6, style="italic")

    # ============ Skip path (THE KEY DIFFERENCE) ============
    y_skip_top = y_raw + 1.0
    # Exit raw X from the top
    if show_skip:
        arrow(ax, (x_x, y_raw + 0.42), (x_x, y_skip_top),
              color=C_SKIP, lw=2.2)
        arrow(ax, (x_x, y_skip_top), (x_merge, y_skip_top),
              color=C_SKIP, lw=2.2)
        arrow(ax, (x_merge, y_skip_top), (x_merge, y_row + 0.7),
              color=C_SKIP, lw=2.2)
        label(ax, (x_x + x_merge) / 2, y_skip_top + 0.35,
              r"\textbf{skip: raw $\mathbf{X}$ at visible cells ($\mathbf{M}{=}0$)}",
              color=C_SKIP, fontsize=9, weight="bold", style="normal")
    else:
        # Dashed gray, ends in an X
        arrow(ax, (x_x, y_raw + 0.42), (x_x, y_skip_top),
              color="#999", lw=1.0, linestyle="--")
        arrow(ax, (x_x, y_skip_top), (x_merge, y_skip_top),
              color="#999", lw=1.0, linestyle="--")
        # the X at the end (above the merge box)
        x_mark_y = y_row + 0.85
        ax.plot([x_merge - 0.15, x_merge + 0.15], [x_mark_y - 0.15, x_mark_y + 0.15],
                color="#999", lw=1.5)
        ax.plot([x_merge - 0.15, x_merge + 0.15], [x_mark_y + 0.15, x_mark_y - 0.15],
                color="#999", lw=1.5)
        # stop the dashed line above the X
        arrow(ax, (x_merge, y_skip_top), (x_merge, x_mark_y + 0.2),
              color="#999", lw=1.0, linestyle="--")
        label(ax, (x_x + x_merge) / 2, y_skip_top + 0.35,
              r"\textbf{skip NOT wired in v1}",
              color="#999", fontsize=9, weight="bold", style="normal")

    # ============ Result badge (placed clearly below the masked row) ============
    y_badge = y_mask - 1.6
    if show_skip:
        box(ax, 7.5, y_badge, 11, 0.6,
            r"v2: visible-cell ToA RMSE = 0.00 m" + "\n" +
            r"$\mathbf{X}$ passes through unchanged",
            fc=C_RESULT_GOOD, ec="#0d6b3b",
            text_color="#0d6b3b", fontsize=8, weight="bold")
    else:
        box(ax, 7.5, y_badge, 11, 0.6,
            r"v1: visible-cell ToA RMSE = 1.78 m" + "\n" +
            r"encoder denoises the raw measurements",
            fc=C_RESULT_BAD, ec="#a52a2a",
            text_color="#a52a2a", fontsize=8, weight="bold")

    # ============ Panel label (rotated, on the left) ============
    label(ax, -1.6, y_row, panel_label,
          color=panel_label_color, fontsize=12, weight="bold", style="normal")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-2, 17)
    ax.set_ylim(-7, 14)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    fig.suptitle("SSL head: v1 (no skip) vs v2 (with hard skip-connection)",
                  fontsize=13, fontweight="bold", y=0.97)
    fig.text(0.5, 0.93,
              r"Red path is the new piece — solid and wired in v2, dashed with $\times$ (inactive) in v1.",
              ha="center", fontsize=9, color="#555", style="italic")

    # Two panels
    draw_panel(ax, yc=9.0, show_skip=False,
                panel_label="v1:\nNO\nskip", panel_label_color="#888")
    # divider
    ax.plot([-2, 16], [4.0, 4.0], color="#bbb", linestyle=":", lw=1.0)
    draw_panel(ax, yc=-0.5, show_skip=True,
                panel_label="v2:\nWITH\nskip", panel_label_color=C_SKIP)

    # Bottom: what stays the same
    box(ax, 7.5, -6.0, 14, 0.55,
        r"Same loss, same encoder weights \textemdash\ the skip is an inference-time architectural choice.",
        fc="#e6f0fa", ec="#1a4d8c", fontsize=9, text_color="#1a4d8c",
        weight="bold")

    out_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent.parent
        / "figures" / "tier0_ssl_head_skip_diagram.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
