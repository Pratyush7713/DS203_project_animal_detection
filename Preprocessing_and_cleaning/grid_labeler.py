'''#!/usr/bin/env python3
"""
Grid label CSV generator (8x8 over 800x600 images).

- Expects a folder of original/overlay images (any readable format) and a matching
  folder of binary segmentation masks with the *same filenames* (stem).
- For each image, the corresponding mask is split into an 8x8 grid.
- A cell is labeled 1 if the number of non‑zero pixels in that cell > MIN_MASK_PIX,
  else 0. (Default MIN_MASK_PIX=10, per your guideline.)
- Outputs a CSV with 65 columns: [image, c01, ..., c64].
- If an image has no matching mask and --skip_missing_mask is NOT set, the row is
  written with all zeros; otherwise it's skipped.

You can adapt paths by passing command‑line flags, or hardcode defaults below.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

# ---------------------- defaults (edit as needed) ----------------------
DEFAULT_IMAGE_DIR = "/home/pratyush/Desktop/Overlay"   # e.g., ".../Overlay"
DEFAULT_MASK_DIR  = "/home/pratyush/Desktop/Mask"                  # e.g., ".../Mask"
DEFAULT_OUT_CSV   = "/home/pratyush/Desktop/DS_Project/data"
TARGET_W, TARGET_H = 800, 600
GRID_W, GRID_H     = 8, 8
MIN_MASK_PIX       = 10  # threshold: if >10 nonzero pixels in a cell -> label 1


# ---------------------------- helpers ----------------------------------
def list_images(dir_path: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() in exts])


def load_mask_bool(path: Path, size: Tuple[int, int]) -> np.ndarray:
    """Load mask image -> boolean array (H,W). Resizes to size (W,H) if needed."""
    img = Image.open(path)
    if img.mode not in ("1", "L"):
        # If colored mask, convert to grayscale first
        img = img.convert("L")
    # Resize to target with nearest neighbor to keep hard edges
    if img.size != size:
        img = img.resize(size, resample=Image.NEAREST)
    arr = np.array(img)
    # Anything >0 counts as mask
    return arr > 0


def grid_labels_from_mask(mask_bool: np.ndarray,
                          grid_w: int = GRID_W,
                          grid_h: int = GRID_H,
                          min_pix: int = MIN_MASK_PIX) -> List[int]:
    """Return list of 64 ints (row-major order 1..64)"""
    H, W = mask_bool.shape
    cell_w = W // grid_w
    cell_h = H // grid_h

    labels: List[int] = []
    for r in range(grid_h):
        y0 = r * cell_h
        y1 = (r + 1) * cell_h
        for c in range(grid_w):
            x0 = c * cell_w
            x1 = (c + 1) * cell_w
            cell = mask_bool[y0:y1, x0:x1]
            count = int(np.count_nonzero(cell))
            labels.append(1 if count > min_pix else 0)
    return labels


def main():
    ap = argparse.ArgumentParser(description="Generate 8x8 grid CSV from Mask images.")
    ap.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR, help="Folder with original/overlay images (names used in first column).")
    ap.add_argument("--mask_dir",  default=DEFAULT_MASK_DIR,  help="Folder with binary masks (same filenames).")
    ap.add_argument("--out_csv",   default=DEFAULT_OUT_CSV,   help="Output CSV path.")
    ap.add_argument("--target_w", type=int, default=TARGET_W)
    ap.add_argument("--target_h", type=int, default=TARGET_H)
    ap.add_argument("--grid_w",   type=int, default=GRID_W)
    ap.add_argument("--grid_h",   type=int, default=GRID_H)
    ap.add_argument("--min_pix",  type=int, default=MIN_MASK_PIX, help=">min_pix nonzero pixels => 1")
    ap.add_argument("--skip_missing_mask", action="store_true", help="If set, rows without a matching mask are skipped.")
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    mask_dir  = Path(args.mask_dir)
    out_csv   = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise SystemExit(f"Image dir not found: {image_dir}")
    if not mask_dir.exists():
        raise SystemExit(f"Mask dir not found: {mask_dir}")

    images = list_images(image_dir)
    if not images:
        raise SystemExit(f"No images found in {image_dir}")

    # Build a lookup for mask files by stem
    mask_lookup = {p.stem: p for p in list_images(mask_dir)}

    headers = ["image"] + [f"c{i:02d}" for i in range(1, args.grid_w * args.grid_h + 1)]

    rows_written = 0
    skipped = 0
    zeros_due_to_missing = 0

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for img_path in images:
            stem = img_path.stem
            mask_path = mask_lookup.get(stem)

            if mask_path is None:
                if args.skip_missing_mask:
                    skipped += 1
                    continue
                # write zeros for all 64 cells
                labels = [0] * (args.grid_w * args.grid_h)
                zeros_due_to_missing += 1
            else:
                mask_bool = load_mask_bool(mask_path, size=(args.target_w, args.target_h))
                # sanity: ensure expected 600x800
                H, W = mask_bool.shape
                if (W, H) != (args.target_w, args.target_h):
                    # We resized inside load_mask_bool, so this shouldn't happen
                    pass
                labels = grid_labels_from_mask(mask_bool, args.grid_w, args.grid_h, args.min_pix)

            row = [img_path.name] + labels
            writer.writerow(row)
            rows_written += 1

    print(f"Wrote {rows_written} rows to {out_csv}")
    if skipped:
        print(f"Skipped {skipped} images without matching masks (per --skip_missing_mask).")
    if zeros_due_to_missing:
        print(f"{zeros_due_to_missing} rows used zeros because mask was missing.")

if __name__ == "__main__":
    main()
'''


#!/usr/bin/env python3
"""
Generate 8x8 grid CSV and visual overlay for animal detection.

Folders expected:
- Actual: original images (names used in CSV)
- Mask: binary masks (same filename as actual image)
Output:
- labels.csv (65 columns)
- Grid overlays saved in output folder
"""

import os
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

# ---------------- CONFIG ----------------
ACTUAL_DIR = "/home/pratyush/Desktop/DS_Project/data/preprocessed"
MASK_DIR   = "/home/pratyush/Desktop/Mask"
OUT_CSV    = "/home/pratyush/Desktop/DS_Project/outputs/lebal_csv.csv"
OUT_GRID   = "/home/pratyush/Desktop/DS_Project/outputs/lebal_image"

TARGET_W, TARGET_H = 800, 600
GRID_W, GRID_H = 8, 8
CELL_W, CELL_H = TARGET_W // GRID_W, TARGET_H // GRID_H
THRESHOLD = 10  # >10 mask pixels = animal present

# ----------------------------------------
def list_images(folder):
    return sorted([f for f in Path(folder).iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}])

def load_mask(path):
    img = Image.open(path).convert("L").resize((TARGET_W, TARGET_H), resample=Image.NEAREST)
    return np.array(img) > 0

def draw_grid_overlay(image_path, labels):
    img = Image.open(image_path).convert("RGB").resize((TARGET_W, TARGET_H))
    draw = ImageDraw.Draw(img)
    
    # Darken cells with 1s
    for i, val in enumerate(labels):
        if val == 1:
            row = i // GRID_W
            col = i % GRID_W
            x0, y0 = col * CELL_W, row * CELL_H
            x1, y1 = x0 + CELL_W, y0 + CELL_H
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(overlay)
            d.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 90))
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    
    # Draw yellow grid lines
    for x in range(0, TARGET_W + 1, CELL_W):
        draw.line([(x, 0), (x, TARGET_H)], fill=(255, 255, 0), width=2)
    for y in range(0, TARGET_H + 1, CELL_H):
        draw.line([(0, y), (TARGET_W, y)], fill=(255, 255, 0), width=2)
    
    # Label cell numbers
    for i in range(GRID_W * GRID_H):
        row = i // GRID_W
        col = i % GRID_W
        x, y = col * CELL_W + 5, row * CELL_H + 5
        draw.text((x, y), str(i + 1), fill=(255, 255, 0))
    
    os.makedirs(OUT_GRID, exist_ok=True)
    out_path = Path(OUT_GRID) / Path(image_path).name
    img.save(out_path)

def main():
    os.makedirs(Path(OUT_CSV).parent, exist_ok=True)
    images = list_images(ACTUAL_DIR)
    mask_lookup = {p.stem: p for p in list_images(MASK_DIR)}

    header = ["image"] + [f"c{i+1:02d}" for i in range(64)]
    rows = []

    for img_path in images:
        name = Path(img_path).name
        mask_path = mask_lookup.get(Path(img_path).stem)
        if not mask_path:
            labels = [0] * 64
        else:
            mask = load_mask(mask_path)
            labels = []
            for r in range(GRID_H):
                for c in range(GRID_W):
                    x0, y0 = c * CELL_W, r * CELL_H
                    x1, y1 = x0 + CELL_W, y0 + CELL_H
                    cell = mask[y0:y1, x0:x1]
                    labels.append(1 if np.count_nonzero(cell) > THRESHOLD else 0)
            draw_grid_overlay(img_path, labels)
        rows.append([name] + labels)
    
    # Write CSV
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"✅ CSV saved to: {OUT_CSV}")
    print(f"✅ Grid visuals saved to: {OUT_GRID}")

if __name__ == "__main__":
    main()
