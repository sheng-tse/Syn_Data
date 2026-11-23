"""
Synthetic data generator for images.

Each output image gets two kinds of variation:
  1. Global jitter  - overall exposure, contrast, and color/white-balance shifts
  2. Local brightness - random regions brightened/darkened
       mode "regions": random rectangles/ellipses with feathered (blurred) edges
       mode "blobs":   smooth brightness field from random Gaussian blobs (more natural)

Setup:
  pip install opencv-python numpy

Usage:
  python regional_brightness.py --input ./images --output ./synthetic --n 5 --mode blobs
  python regional_brightness.py --input ./images --output ./synthetic --no-global   # local only
"""
import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Global variation
# ---------------------------------------------------------------------------
def srgb_to_linear(x):
    """sRGB-encoded values in [0, 1] -> linear light."""
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    """Linear light -> sRGB-encoded values in [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def global_jitter(img, rng, exposure=(0.7, 1.3), contrast=(0.8, 1.2), max_wb=0.1, max_tint=0.0):
    """Change overall exposure, white balance and contrast.
    Exposure and white balance are applied in linear light, the way a camera
    applies them; contrast acts on the encoded values, like an editor's slider.
    White balance moves along the blue-amber axis (+ warmer, - cooler), with an
    optional green-magenta tint (off by default).
    img: BGR or grayscale uint8. Returns uint8."""
    lin = srgb_to_linear(img.astype(np.float32) / 255.0)
    lin *= rng.uniform(*exposure)                          # exposure
    if lin.ndim == 3:
        shift = rng.uniform(-max_wb, max_wb)               # + warmer, - cooler
        tint = rng.uniform(-max_tint, max_tint) if max_tint > 0 else 0.0
        lin[..., 0] *= 1.0 - shift                         # blue (OpenCV order is B, G, R)
        lin[..., 2] *= 1.0 + shift                         # red
        lin[..., 1] *= 1.0 + tint                          # green
    out = (linear_to_srgb(lin) * 255.0).astype(np.float32)
    mean = out.mean()
    out = (out - mean) * rng.uniform(*contrast) + mean     # contrast
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Local (regional) brightness variation
# ---------------------------------------------------------------------------
def _region_field(h, w, rng, n_regions, factor_range, feather):
    """Brightness multiplier map from random rect/ellipse regions with soft edges."""
    field = np.ones((h, w), np.float32)
    for _ in range(n_regions):
        mask = np.zeros((h, w), np.float32)
        rw = int(rng.uniform(0.1, 0.5) * w)
        rh = int(rng.uniform(0.1, 0.5) * h)
        cx, cy = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        if rng.random() < 0.5:
            cv2.rectangle(mask, (cx - rw // 2, cy - rh // 2),
                          (cx + rw // 2, cy + rh // 2), 1.0, -1)
        else:
            angle = rng.uniform(0, 180)
            cv2.ellipse(mask, (cx, cy), (max(rw // 2, 1), max(rh // 2, 1)),
                        angle, 0, 360, 1.0, -1)
        k = int(max(h, w) * feather) | 1  # Gaussian kernel must be odd
        if k > 1:
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        factor = rng.uniform(*factor_range)
        field *= 1.0 + mask * (factor - 1.0)
    return field


def _blob_field(h, w, rng, n_regions, factor_range):
    """Smooth brightness field from a sum of random Gaussian blobs."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field = np.zeros((h, w), np.float32)
    for _ in range(n_regions):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        sigma = rng.uniform(0.08, 0.3) * max(h, w)
        amp = rng.uniform(*factor_range) - 1.0  # >0 brighter, <0 darker
        field += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return 1.0 + field


def random_regional_brightness(img, rng, n_regions=(2, 5), factor_range=(0.5, 1.5),
                               mode="regions", feather=0.08):
    """Brighten/darken random regions. img: BGR or grayscale uint8. Returns uint8.
    Overlapping regions multiply (or blobs add), so the combined field is held to
    factor_range: no spot ends up darker or brighter than a single region allows."""
    h, w = img.shape[:2]
    n = int(rng.integers(n_regions[0], n_regions[1] + 1))

    if mode == "regions":
        field = _region_field(h, w, rng, n, factor_range, feather)
    elif mode == "blobs":
        field = _blob_field(h, w, rng, n, factor_range)
    else:
        raise ValueError(f"unknown mode: {mode}")
    field = np.clip(field, max(0.0, factor_range[0]), factor_range[1])

    if img.ndim == 2:  # grayscale
        return np.clip(img.astype(np.float32) * field, 0, 255).astype(np.uint8)

    # adjust only the V channel so the local change is a pure brightness change
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * field, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def image_seed(seed, name):
    """Per-image seed derived from the run seed and the file name, so an image's
    variants don't depend on which other files are in the folder. None = random."""
    if seed is None:
        return None
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def augment(img, rng, args):
    out = img
    if not args.no_global:
        out = global_jitter(
            out, rng,
            exposure=(args.min_exposure, args.max_exposure),
            contrast=(args.min_contrast, args.max_contrast),
            max_wb=args.max_wb, max_tint=args.max_tint,
        )
    out = random_regional_brightness(
        out, rng,
        n_regions=(args.min_regions, args.max_regions),
        factor_range=(args.min_factor, args.max_factor),
        mode=args.mode, feather=args.feather,
    )
    return out


def main():
    p = argparse.ArgumentParser(description="Generate synthetic images with global + regional brightness variation.")
    p.add_argument("--input", required=True, help="folder of input images")
    p.add_argument("--output", required=True, help="folder for synthetic images")
    p.add_argument("--n", type=int, default=5, help="variants per image")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")

    # local brightness
    p.add_argument("--mode", choices=["regions", "blobs"], default="regions")
    p.add_argument("--min-regions", type=int, default=2)
    p.add_argument("--max-regions", type=int, default=5)
    p.add_argument("--min-factor", type=float, default=0.5, help="darkest regional multiplier")
    p.add_argument("--max-factor", type=float, default=1.5, help="brightest regional multiplier")
    p.add_argument("--feather", type=float, default=0.08, help="edge softness (regions mode)")

    # global jitter
    p.add_argument("--no-global", action="store_true", help="disable global jitter")
    p.add_argument("--min-exposure", type=float, default=0.7)
    p.add_argument("--max-exposure", type=float, default=1.3)
    p.add_argument("--min-contrast", type=float, default=0.8)
    p.add_argument("--max-contrast", type=float, default=1.2)
    p.add_argument("--max-wb", type=float, default=0.1,
                   help="largest warm/cool white-balance shift (0.1 = red and blue gains of 0.9-1.1, in opposite directions)")
    p.add_argument("--max-tint", type=float, default=0.0, help="largest green-magenta tint (0 = off)")
    args = p.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    if not in_dir.is_dir():
        raise SystemExit(f"input folder not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [f for f in sorted(in_dir.iterdir()) if f.suffix.lower() in exts]
    if not files:
        raise SystemExit(f"no images found in {in_dir}")

    for path in files:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"skip (unreadable): {path.name}")
            continue
        if img.dtype != np.uint8:  # e.g. 16-bit PNG/TIFF
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        alpha = None
        if img.ndim == 3 and img.shape[2] == 4:  # keep alpha channel untouched
            img, alpha = img[..., :3], img[..., 3]

        rng = np.random.default_rng(image_seed(args.seed, path.name))
        for i in range(args.n):
            aug = augment(img, rng, args)
            if alpha is not None:
                aug = np.dstack([aug, alpha])
            cv2.imwrite(str(out_dir / f"{path.stem}_syn{i}{path.suffix}"), aug)
        print(f"done: {path.name} -> {args.n} variants")

    print(f"finished: {len(files) * args.n} images written to {out_dir}")


if __name__ == "__main__":
    main()