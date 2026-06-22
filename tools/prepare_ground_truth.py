#!/usr/bin/env python3
"""
prepare_ground_truth.py — turn hand-drawn tracings into evaluable label masks.

The GraFT evaluation expects a folder of per-filament binary masks named
``label1.png``, ``label2.png``, … with a **black (0) background** and the
filament as non-zero pixels (see ``utilsGraFT.prepare_ground_truth_layers``).
Hand tracing on a tablet, however, usually produces the opposite: strokes drawn
on a **white** canvas, often in colour, exported with cryptic names — and
sometimes every filament squeezed onto a single multi-colour image.

This tool converts either form into the expected ``label<N>.png`` masks:

  • ``layers``    — one filament per input image (the robust, recommended path).
                    Background polarity is auto-detected, so it works for
                    dark-on-white *and* white-on-dark exports. Empty layers are
                    skipped. This is colour-agnostic: each layer's stroke becomes
                    the mask regardless of the pen colour, which sidesteps the
                    "light colours are hard to extract" problem entirely.

  • ``composite`` — a single image with every filament in a different colour
                    (best-effort). Filaments are separated by hue. Overlapping
                    strokes cannot be disentangled reliably, so prefer ``layers``
                    whenever the per-layer exports still exist.

Examples
--------
Per-layer exports (recommended)::

    python tools/prepare_ground_truth.py layers \\
        --input  "raw_layers/" \\
        --output "data_samples/biological/ground_truth" \\
        --preview

Single multi-colour composite (best-effort)::

    python tools/prepare_ground_truth.py composite \\
        --input  "labels_colored.png" \\
        --output "ground_truth" --n-colors 6 --preview

Output masks are 8-bit PNG (0 = background, 255 = filament), named
``label1.png … labelN.png`` and ready to pass to the workflow via ``--gt``.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys

import warnings

import numpy as np
from skimage import color, io, img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects

# remove_small_objects emits a FutureWarning about a parameter rename; the call
# is correct for current skimage, so keep the output clean.
warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("prepare_gt")

IMG_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _natural_key(name: str):
    """Sort key that orders embedded numbers numerically (layer2 < layer10)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def to_grayscale_uint8(img: np.ndarray) -> np.ndarray:
    """Collapse any image (RGBA / RGB / gray, float or int) to uint8 grayscale.

    For RGBA inputs whose alpha channel actually carries information (a stroke
    drawn on a transparent canvas), the alpha channel *is* the mask, so we use
    it directly. Otherwise the alpha is dropped and RGB is converted to gray.
    """
    if img.ndim == 3 and img.shape[-1] == 4:
        alpha = img[..., 3]
        if 0 < alpha.min() or alpha.max() == alpha.min():
            # Alpha is constant (fully opaque) → uninformative; drop it.
            img = img[..., :3]
        else:
            # Transparent background: opaque pixels are the drawing.
            return img_as_ubyte(alpha / (alpha.max() or 1))
    if img.ndim == 3 and img.shape[-1] == 3:
        img = color.rgb2gray(img)
    if img.ndim == 3:  # defensive: any other channel count → mean
        img = img.mean(axis=-1)
    if img.dtype != np.uint8:
        img = img_as_ubyte(img / img.max()) if img.max() > 1.0 else img_as_ubyte(img)
    return img


def detect_bg_bright(gray: np.ndarray) -> bool:
    """True if the image has a bright (white) background, judged from its border."""
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    return bool(np.median(border) > 127)


def foreground_from_gray(gray: np.ndarray, threshold: int | None,
                         use_otsu: bool, min_pixels: int) -> np.ndarray:
    """Binary foreground mask from a grayscale image.

    Background polarity is detected from the border: if the frame edge is bright
    the canvas is white, so the image is inverted to make strokes bright before
    thresholding. Small speckles (compression noise, stray dots) are removed.
    """
    bg_is_bright = detect_bg_bright(gray)
    work = (255 - gray) if bg_is_bright else gray  # strokes now bright-on-dark

    if use_otsu:
        try:
            t = threshold_otsu(work)
        except ValueError:
            t = 15
    else:
        t = 15 if threshold is None else threshold

    mask = work > t
    if min_pixels > 0 and mask.any():
        mask = remove_small_objects(mask, min_size=min_pixels)
    return mask


def _save_mask(mask: np.ndarray, out_path: str) -> None:
    io.imsave(out_path, (mask.astype(np.uint8) * 255), check_contrast=False)


def _write_preview(masks: list[np.ndarray], out_dir: str) -> None:
    """Save a coloured overlay of all labels for a quick visual sanity check."""
    if not masks:
        return
    labelled = np.zeros(masks[0].shape, dtype=np.int32)
    for i, m in enumerate(masks, start=1):
        labelled[m] = i
    rgb = color.label2rgb(labelled, bg_label=0)
    io.imsave(os.path.join(out_dir, "_preview_labels.png"),
              img_as_ubyte(rgb), check_contrast=False)
    log.info("Wrote preview → %s", os.path.join(out_dir, "_preview_labels.png"))


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run_layers(args: argparse.Namespace) -> int:
    files = sorted(
        (f for f in os.listdir(args.input)
         if f.lower().endswith(IMG_EXTS) and os.path.isfile(os.path.join(args.input, f))),
        key=_natural_key,
    )
    if not files:
        log.error("No images found in '%s'.", args.input)
        return 1
    log.info("Found %d candidate layer file(s) in %s", len(files), args.input)

    os.makedirs(args.output, exist_ok=True)
    masks, idx = [], 0
    for fname in files:
        gray = to_grayscale_uint8(io.imread(os.path.join(args.input, fname)))
        if args.require_background != "auto":
            want_bright = (args.require_background == "white")
            if detect_bg_bright(gray) != want_bright:
                log.info("  skip %-40s (background is %s; --require-background=%s)",
                         fname, "white" if not want_bright else "dark", args.require_background)
                continue
        mask = foreground_from_gray(gray, args.threshold, args.otsu, args.min_pixels)
        n = int(mask.sum())
        frac = n / mask.size
        if n < args.min_pixels:
            log.info("  skip %-40s (empty: %d px)", fname, n)
            continue
        if frac > args.max_fraction:
            log.warning("  skip %-40s (too dense: %.0f%% foreground — looks like a "
                        "source image, not a single-filament layer)", fname, 100 * frac)
            continue
        idx += 1
        out = os.path.join(args.output, f"label{idx}.png")
        _save_mask(mask, out)
        masks.append(mask)
        log.info("  %-40s → label%d.png (%d px)", fname, idx, n)

    if idx == 0:
        log.error("No usable layers produced — check --threshold / --min-pixels.")
        return 1
    log.info("Wrote %d label mask(s) to %s", idx, args.output)
    if args.preview:
        _write_preview(masks, args.output)
    return 0


def run_composite(args: argparse.Namespace) -> int:
    img = io.imread(args.input)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    if img.ndim != 3 or img.shape[-1] != 3:
        log.error("Composite mode expects a colour (RGB) image.")
        return 1
    rgb = img_as_ubyte(img / img.max()) if img.max() > 1.0 else img_as_ubyte(img)

    gray = color.rgb2gray(rgb)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg_bright = np.median(border) > 0.5
    # Foreground = pixels that are coloured (have saturation) or differ from bg.
    hsv = color.rgb2hsv(rgb)
    sat, val = hsv[..., 1], hsv[..., 2]
    fg = (sat > 0.15) & ((val < 0.95) if bg_bright else (val > 0.05))
    fg = remove_small_objects(fg, min_size=args.min_pixels)
    if not fg.any():
        log.error("No coloured foreground detected in composite.")
        return 1

    hue = hsv[..., 0][fg]
    hist, edges = np.histogram(hue, bins=180, range=(0, 1))
    # Pick the N most populated hue bins as filament colours.
    top = np.argsort(hist)[::-1]
    centers = []
    for b in top:
        h = (edges[b] + edges[b + 1]) / 2
        if all(min(abs(h - c), 1 - abs(h - c)) > 0.04 for c in centers):
            centers.append(h)
        if len(centers) >= args.n_colors:
            break

    os.makedirs(args.output, exist_ok=True)
    hue_full = hsv[..., 0]
    masks = []
    for i, h in enumerate(centers, start=1):
        d = np.minimum(np.abs(hue_full - h), 1 - np.abs(hue_full - h))
        m = fg & (d < 0.05)
        m = remove_small_objects(m, min_size=args.min_pixels)
        if m.sum() < args.min_pixels:
            continue
        _save_mask(m, os.path.join(args.output, f"label{i}.png"))
        masks.append(m)
        log.info("  hue %.2f → label%d.png (%d px)", h, i, int(m.sum()))

    log.warning("Composite separation is best-effort: overlapping strokes are NOT "
                "reliably disentangled. Prefer per-layer exports where possible.")
    if args.preview:
        _write_preview(masks, args.output)
    return 0 if masks else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Convert hand-drawn tracings into label<N>.png masks for GraFT evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    pl = sub.add_parser("layers", help="One filament per input image (recommended).")
    pl.add_argument("--input", required=True, help="Folder of per-layer tracings.")
    pl.add_argument("--output", required=True, help="Folder to write label<N>.png into.")
    pl.add_argument("--threshold", type=int, default=None,
                    help="Fixed threshold (0-255) on the polarity-corrected image. "
                         "Lower catches fainter strokes. Default 15.")
    pl.add_argument("--otsu", action="store_true",
                    help="Use Otsu's threshold instead of a fixed value.")
    pl.add_argument("--min-pixels", type=int, default=20,
                    help="Drop masks / speckles smaller than this many pixels.")
    pl.add_argument("--max-fraction", type=float, default=0.4,
                    help="Skip inputs whose foreground exceeds this fraction "
                         "(filters out source images mixed into the layer folder).")
    pl.add_argument("--require-background", choices=["auto", "white", "dark"],
                    default="auto",
                    help="Only process images with this background polarity. Set "
                         "'white' to keep hand-drawn-on-white tracings and skip "
                         "dark microscopy source frames sharing the folder.")
    pl.add_argument("--preview", action="store_true",
                    help="Also write a coloured _preview_labels.png overlay.")
    pl.set_defaults(func=run_layers)

    pc = sub.add_parser("composite", help="Single multi-colour image (best-effort).")
    pc.add_argument("--input", required=True, help="Path to the multi-colour image.")
    pc.add_argument("--output", required=True, help="Folder to write label<N>.png into.")
    pc.add_argument("--n-colors", type=int, default=6, help="Number of filament colours to separate.")
    pc.add_argument("--min-pixels", type=int, default=20, help="Drop masks smaller than this.")
    pc.add_argument("--preview", action="store_true", help="Write a coloured overlay preview.")
    pc.set_defaults(func=run_composite)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
