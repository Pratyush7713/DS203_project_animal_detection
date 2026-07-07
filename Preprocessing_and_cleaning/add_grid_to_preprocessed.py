# --- Add 8×8 grid + numbering (bottom-right) on all preprocessed images ---

import os
import cv2
from glob import glob

# Paths (change if needed)
BASE = "/home/pratyush/Desktop/DS_Project"
IN_DIR  = f"{BASE}/data/preprocessed"
OUT_DIR = f"{BASE}/data/after_grid_image_preprocessed"
os.makedirs(OUT_DIR, exist_ok=True)

# Grid constants
TARGET_W, TARGET_H = 800, 600
GRID_W, GRID_H = 8, 8
CELL_W, CELL_H = TARGET_W // GRID_W, TARGET_H // GRID_H  # 100x75

# Drawing styles
LINE_COLOR = (0, 255, 255)   # yellow (BGR in OpenCV)
LINE_THICK = 1
TEXT_COLOR = (0, 255, 255)   # yellow
TEXT_SHADOW = (0, 0, 0)      # black outline for contrast
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_THICK = 1
MARGIN = 4                   # px padding from cell edge

def draw_grid_and_numbers(img_bgr):
    # Ensure size 800x600 (your preprocessed should already be)
    h, w = img_bgr.shape[:2]
    if (w, h) != (TARGET_W, TARGET_H):
        img_bgr = cv2.resize(img_bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

    # 1) draw vertical lines
    for c in range(GRID_W + 1):
        x = c * CELL_W
        cv2.line(img_bgr, (x, 0), (x, TARGET_H), LINE_COLOR, LINE_THICK)

    # 2) draw horizontal lines
    for r in range(GRID_H + 1):
        y = r * CELL_H
        cv2.line(img_bgr, (0, y), (TARGET_W, y), LINE_COLOR, LINE_THICK)

    # 3) number cells 1..64 (row-major), bottom-right corner of each cell
    idx = 1
    for r in range(GRID_H):
        for c in range(GRID_W):
            x0, y0 = c * CELL_W, r * CELL_H
            # bottom-right anchor (we’ll right-align using text size)
            label = f"{idx}"
            (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
            x_text = x0 + CELL_W - tw - MARGIN
            y_text = y0 + CELL_H - MARGIN  # OpenCV uses baseline at bottom

            # shadow (outline) for readability
            cv2.putText(img_bgr, label, (x_text+1, y_text+1), FONT, FONT_SCALE, TEXT_SHADOW, FONT_THICK+1, cv2.LINE_AA)
            # main text
            cv2.putText(img_bgr, label, (x_text, y_text), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICK, cv2.LINE_AA)
            idx += 1

    return img_bgr

# Process all images
exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
files = []
for e in exts:
    files.extend(glob(os.path.join(IN_DIR, e)))
files.sort()

processed, skipped = 0, 0
for fpath in files:
    img = cv2.imread(fpath)
    if img is None:
        print(f"⚠️ Could not read: {os.path.basename(fpath)}"); skipped += 1; continue

    out_img = draw_grid_and_numbers(img)
    out_path = os.path.join(OUT_DIR, os.path.basename(fpath))
    # Save as same format as input (PNG/JPG/etc.)
    ok = cv2.imwrite(out_path, out_img)
    if not ok:
        print(f"⚠️ Failed to save: {out_path}"); skipped += 1
    else:
        processed += 1

print(f"✅ Done. Saved {processed} images to {OUT_DIR}. Skipped {skipped}.")
