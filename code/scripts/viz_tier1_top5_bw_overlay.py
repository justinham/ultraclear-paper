"""Add a black/white-print-friendly pattern overlay to an existing
`tier1_top5_per_env.png` figure.

The original figure encodes per-point localization error via the
`coolwarm` colormap (blue=low, red=high). When printed in grayscale,
both ends of `coolwarm` collapse to similar mid-grays, so the error
information is lost.

This script keeps every original pixel of the figure (colors,
position, axis labels, colorbar — nothing changes) and ONLY adds a
stippling pattern whose density scales with the same per-point error
that the color encodes. The pattern rides ON TOP of the dots:
   - low-error (blue) dots get sparse stippling  -> light gray in B&W
   - high-error (red) dots get dense stippling   -> dark in B&W
On a color print, the colors still tell the story. On a B&W laser
print, the stippling density carries the same story.

The script is data-agnostic: it never touches the underlying CSV or
checkpoint, only the PNG pixels.

Usage:
    python3.11 -m scripts.viz_tier1_top5_bw_overlay \\
        --in  figures/tier1_top5_per_env.png \\
        --out figures/tier1_top5_per_env_bw.png
"""
from __future__ import annotations
import argparse
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def _coolwarm_to_err(rgb: np.ndarray, vmin: float = 0.0, vmax: float = 0.7) -> np.ndarray:
    """Map an RGB color in matplotlib's `coolwarm` colormap back to a
    fractional error in [0, 1]. coolwarm is a diverging colormap with
    three control points:
       low (err=0):     (68,  90, 200)   — cool blue
       mid (err=0.5):   (200, 200, 200)  — neutral gray
       high (err=1):    (200,  60, 50)   — warm red
    We use signed distance from the mid-gray axis (red-blue) so the
    mapping is monotonic across the whole colormap."""
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    # "warmth" axis: positive = red side, negative = blue side
    warmth = (r - b) / 255.0  # roughly in [-1, 1]
    # Normalize: blue end -> warmth≈-0.52, red end -> warmth≈+0.59
    warmth_norm = (warmth + 0.52) / (0.59 + 0.52)
    warmth_norm = np.clip(warmth_norm, 0.0, 1.0)
    err = vmin + warmth_norm * (vmax - vmin)
    return err


def _is_data_pixel(rgb: np.ndarray) -> np.ndarray:
    """Boolean mask: True where this pixel is a scatter dot (vs.
    background, axis chrome, or colorbar). A data pixel is saturated
    and lies somewhere on the coolwarm trajectory between blue and red."""
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    is_colored = chroma > 15
    not_chrome = luma > 25
    return is_colored & not_chrome


def _is_tab10_outline(rgb: np.ndarray) -> np.ndarray:
    """matplotlib's tab10 colormap has 10 saturated colors used for
    the auto-cycled marker edgecolors. They're NOT part of the
    coolwarm error encoding — exclude them from stippling."""
    tab10 = {(31, 119, 180), (255, 127, 14), (44, 160, 44),
             (214, 39, 40), (148, 103, 189), (140, 86, 75),
             (227, 119, 194), (127, 127, 127), (188, 189, 34),
             (23, 190, 207)}
    r, g, b = rgb[..., 0].astype(np.int32), rgb[..., 1].astype(np.int32), rgb[..., 2].astype(np.int32)
    # Match tab10 within ±2 per channel (anti-alias edges)
    matched = np.zeros(rgb.shape[:2], dtype=bool)
    for tr, tg, tb in tab10:
        matched |= ((np.abs(r - tr) <= 3) & (np.abs(g - tg) <= 3) & (np.abs(b - tb) <= 3))
    return matched


def _build_cross_overlay(size: tuple[int, int], dot_mask: np.ndarray) -> Image.Image:
    """Draw a small black "+" cross at the center of every scatter dot.

    Simple, high-contrast pattern that reads on B&W print: every dot
    gets a visible black crosshair whether or not the underlying
    color is dark. The pattern density carries no information; the
    cross is uniform across all dots. The dot COLOR still carries the
    error magnitude; on B&W print, the cross just ensures each dot
    is visually distinct from the background.
    """
    H, W = size
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    period = 5
    for y in range(0, H, period):
        for x in range(0, W, period):
            block = dot_mask[y:y+period, x:x+period]
            if not block.any():
                continue
            cy = y + period // 2
            cx = x + period // 2
            # 5x5 black "+": horizontal stroke 5px wide, vertical stroke 5px tall
            for d in range(-2, 3):
                if 0 <= cy < H and 0 <= cx+d < W:
                    overlay[cy, cx+d] = [0, 0, 0, 255]
                if 0 <= cy+d < H and 0 <= cx < W:
                    overlay[cy+d, cx] = [0, 0, 0, 255]
            # Center pixel black
            if 0 <= cy < H and 0 <= cx < W:
                overlay[cy, cx] = [0, 0, 0, 255]
    return Image.fromarray(overlay, mode="RGBA")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path",
                   default="../figures/tier1_top5_per_env.png")
    p.add_argument("--out", dest="out_path",
                   default="../figures/tier1_top5_per_env_bw.png")
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=0.7)
    args = p.parse_args()

    src = Image.open(args.in_path).convert("RGB")
    arr = np.array(src)
    H, W, _ = arr.shape
    print(f"loaded {args.in_path}: {W}x{H}")

    # Mask: only put crosshairs on actual scatter dots (not background,
    # not axis text/lines, not the colorbar, not tab10 outlines).
    is_dot = _is_data_pixel(arr)
    is_outline = _is_tab10_outline(arr)
    is_dot_interior = is_dot & ~is_outline

    overlay = _build_cross_overlay((H, W), is_dot_interior)

    # 4) Composite: crosshairs go ON TOP of the original colors,
    #    but ONLY inside data dots (not over background / chrome).
    stip_arr = np.array(overlay)
    keep_mask = is_dot_interior.astype(np.uint8) * 255
    stip_arr[..., 3] = np.minimum(stip_arr[..., 3], keep_mask)

    src_rgba = src.convert("RGBA")
    composited = Image.alpha_composite(src_rgba, Image.fromarray(stip_arr, mode="RGBA"))
    composited.convert("RGB").save(args.out_path, dpi=(130, 130))
    print(f"saved → {args.out_path}")
    n_stipple = int((stip_arr[..., 3] > 0).sum())
    n_dots = int(is_dot_interior.sum())
    print(f"  stipple dots placed: {n_stipple}  over  data-dot px: {n_dots}")


if __name__ == "__main__":
    main()