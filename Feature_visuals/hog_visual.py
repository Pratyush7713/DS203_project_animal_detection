#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HOG visual demonstration (NumPy 2.0 compatible)
- Original → Grayscale → HOG heatmap → HOG overlay
- Optional: dominant gradient orientation arrows
"""

import os
import warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

warnings.filterwarnings("ignore", message="Unable to import Axes3D")

# ----------- PATHS -----------
BASE = "/home/pratyush/Desktop/DS_Project"
IMG_DIR = f"{BASE}/data/preprocessed"
OUT_DIR = f"{BASE}/outputs/hog_demo"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------- CORE HOG VISUALIZATION -----------
def hog_visualize(img_bgr,
                  orientations=9,
                  pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2),
                  block_norm="L2-Hys",
                  cmap_choice="turbo",
                  overlay_alpha_img=0.6,
                  overlay_alpha_hog=0.6):
    """
    Returns (gray, hog_map_rescaled [0..1], overlay_on_original [BGR])
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Extract HOG features and visualization image
    _, hog_image = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        transform_sqrt=True,
        visualize=True,
        feature_vector=True
    )

    # Normalize safely (NumPy 2.0 compatible)
    hog_image = np.asarray(hog_image, dtype=np.float32)
    hog_norm = (hog_image - np.min(hog_image)) / (np.ptp(hog_image) + 1e-8)

    # Contrast enhancement
    hog_rescaled = exposure.equalize_adapthist(hog_norm, clip_limit=0.03)

    # Apply bright colormap and overlay
    cmap = plt.get_cmap(cmap_choice)
    hog_rgb = (cmap(hog_rescaled)[..., :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img_bgr, overlay_alpha_img, hog_rgb, overlay_alpha_hog, 0)

    return gray, hog_rescaled, overlay

# ----------- ORIENTATION ARROWS -----------
def draw_orientation_arrows(ax, gray, cell=16, stride=16, scale=0.6, color='w'):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi  # [0, π)

    H, W = gray.shape
    xs, ys, us, vs = [], [], [], []

    for y0 in range(0, H - cell + 1, stride):
        for x0 in range(0, W - cell + 1, stride):
            m = mag[y0:y0+cell, x0:x0+cell].ravel()
            a = ang[y0:y0+cell, x0:x0+cell].ravel()
            if m.size == 0: continue

            bins = 9
            hist, edges = np.histogram(a, bins=bins, range=(0, np.pi), weights=m)
            k = int(np.argmax(hist))
            a0 = 0.5 * (edges[k] + edges[k+1])
            u, v = np.cos(a0), np.sin(a0)
            cx, cy = x0 + cell/2, y0 + cell/2
            xs.append(cx); ys.append(cy); us.append(u); vs.append(v)

    ax.quiver(xs, ys, us, vs, angles='xy', scale_units='xy',
              scale=1/scale, color=color, width=0.003,
              headwidth=3, headlength=4)

# ----------- DEMO FUNCTION -----------
def show_and_save(image_path, save_prefix=None, cmap_choice="turbo", show_arrows=True):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"⚠️ Cannot read: {image_path}")
        return

    img_bgr = cv2.resize(img_bgr, (800, 600), interpolation=cv2.INTER_AREA)
    gray, hog_map, overlay = hog_visualize(img_bgr, cmap_choice=cmap_choice)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), dpi=110)
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(gray, cmap="gray"); axes[1].set_title("Grayscale"); axes[1].axis("off")
    im3 = axes[2].imshow(hog_map, cmap=cmap_choice); axes[2].set_title(f"HOG heatmap ({cmap_choice})"); axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); axes[3].set_title("HOG overlay"); axes[3].axis("off")

    if show_arrows:
        draw_orientation_arrows(axes[3], gray, cell=16, stride=16, scale=0.6, color='w')

    plt.tight_layout()
    plt.show()

    if save_prefix:
        base = os.path.join(OUT_DIR, save_prefix)
        cv2.imwrite(base + "_original.jpg", img_bgr)
        cv2.imwrite(base + "_overlay.jpg", overlay)
        plt.imsave(base + "_hogmap.png", hog_map, cmap=cmap_choice)
        print(f"💾 Saved: {OUT_DIR}/{os.path.basename(base)}_*")

# ----------- MAIN EXECUTION -----------
if __name__ == "__main__":
    demo_img = "26102010062.jpg"
    p = os.path.join(IMG_DIR, demo_img)
    if not os.path.exists(p):
        imgs = [f for f in sorted(os.listdir(IMG_DIR)) if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
        if not imgs:
            raise SystemExit(f"No images found in {IMG_DIR}")
        p = os.path.join(IMG_DIR, imgs[0])

    print("Using image:", p)
    show_and_save(p, save_prefix="hog_demo_01", cmap_choice="turbo", show_arrows=True)
