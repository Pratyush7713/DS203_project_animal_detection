#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LBP (Local Binary Patterns) visual demonstration on 8×8 grid:
- Original → Grayscale
- LBP map (uniform, P=8, R=1)
- 8×8 heatmaps: entropy (texture complexity), dominant-bin (pattern index)
- LBP overlay on the image
- Global LBP histogram with top bins
Saves under outputs/lbp_demo/
"""

import os
import warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from scipy.stats import entropy as shannon_entropy  # scipy>=1.6

warnings.filterwarnings("ignore", message="Unable to import Axes3D")

# ----------- PATHS -----------
BASE    = "/home/pratyush/Desktop/DS_Project"
IMG_DIR = f"{BASE}/data/preprocessed"       # your 800×600 set
OUT_DIR = f"{BASE}/outputs/lbp_demo"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------- GRID / GEOMETRY -----------
W, H = 800, 600
GW, GH = 8, 8
CW, CH = W // GW, H // GH  # 100×75

# ----------- LBP CONFIG -----------
P = 8          # points
R = 1          # radius (pixels)
METHOD = "uniform"  # 'uniform' → bins = P + 2
LBP_BINS = P + 2

# ----------- HELPERS -----------
def ensure_size(img_bgr, w=W, h=H):
    if img_bgr.shape[1] != w or img_bgr.shape[0] != h:
        img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    return img_bgr

def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def compute_lbp(gray):
    # skimage returns float image of LBP codes (not normalized)
    lbp = local_binary_pattern(gray, P=P, R=R, method=METHOD)
    return lbp

def lbp_hist(arr, bins=LBP_BINS):
    # values are in [0, P+1] for 'uniform'
    hist, _ = np.histogram(arr, bins=bins, range=(0, bins), density=False)
    return hist.astype(np.float32)

def compute_cell_maps(lbp_img):
    """
    For each 8×8 cell, compute:
      - entropy (Shannon) over LBP histogram
      - dominant-bin (argmax of histogram)
    Returns dict of GH×GW maps.
    """
    ent_map = np.zeros((GH, GW), dtype=np.float32)
    dom_map = np.zeros((GH, GW), dtype=np.float32)

    for r in range(GH):
        for c in range(GW):
            y0, y1 = r*CH, (r+1)*CH
            x0, x1 = c*CW, (c+1)*CW
            block = lbp_img[y0:y1, x0:x1].ravel()
            hist = lbp_hist(block, bins=LBP_BINS)
            p = hist / (hist.sum() + 1e-8)
            ent_map[r, c] = shannon_entropy(p, base=2)  # bits
            dom_map[r, c] = np.argmax(hist)             # 0..(LBP_BINS-1)
    return {"entropy": ent_map, "dominant_bin": dom_map}

def norm01(x):
    x = x.astype(np.float32)
    return (x - np.min(x)) / (np.ptp(x) + 1e-8)

def upsample_to_image(map_2d):
    """Upsample GH×GW map to 600×800 for pretty overlays."""
    m = norm01(map_2d)
    m_img = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return m_img

def overlay_heatmap(img_bgr, norm_map, alpha_img=0.6, alpha_heat=0.6, cmap_name="turbo"):
    cmap = plt.get_cmap(cmap_name)
    heat_rgb = (cmap(norm_map)[..., :3] * 255).astype(np.uint8)[:, :, ::-1]  # RGB→BGR
    return cv2.addWeighted(img_bgr, alpha_img, heat_rgb, alpha_heat, 0)

# ----------- MAIN DEMO -----------
def show_and_save_lbp_demo(image_path, save_prefix="lbp_demo_01"):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"⚠️ Cannot read: {image_path}")
        return

    img_bgr = ensure_size(img_bgr, W, H)
    gray = to_gray(img_bgr)

    # LBP image (float with codes 0..P+1 for 'uniform')
    lbp = compute_lbp(gray)

    # Normalize LBP image to [0,1] for display
    lbp_norm = lbp / (LBP_BINS - 1)

    # Per-cell maps
    maps = compute_cell_maps(lbp)

    # Overlay: use normalized LBP map as heatmap
    overlay = overlay_heatmap(img_bgr, lbp_norm, alpha_img=0.65, alpha_heat=0.55, cmap_name="turbo")

    # ---- FIGURE 1: Original + Grayscale + LBP map + LBP overlay ----
    fig1, axes1 = plt.subplots(1, 4, figsize=(16, 4.8), dpi=110)
    axes1[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); axes1[0].set_title("Original"); axes1[0].axis("off")
    axes1[1].imshow(gray, cmap="gray"); axes1[1].set_title("Grayscale"); axes1[1].axis("off")
    im2 = axes1[2].imshow(lbp_norm, cmap="turbo", vmin=0, vmax=1)
    axes1[2].set_title(f"LBP map (P={P}, R={R}, uniform)"); axes1[2].axis("off")
    plt.colorbar(im2, ax=axes1[2], fraction=0.046, pad=0.04)
    axes1[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); axes1[3].set_title("LBP overlay"); axes1[3].axis("off")
    fig1.tight_layout()
    plt.show()

    # ---- FIGURE 2: 8×8 heatmaps: entropy & dominant-bin ----
    show_keys = ["entropy", "dominant_bin"]
    fig2, axes2 = plt.subplots(1, len(show_keys), figsize=(10, 4.5), dpi=110)
    if len(show_keys) == 1: axes2 = [axes2]
    for i, k in enumerate(show_keys):
        m = maps[k]
        if k == "dominant_bin":
            # scale to 0..1 for colormap; also show discrete ticks later if needed
            disp = m / (LBP_BINS - 1)
            im = axes2[i].imshow(disp, cmap="turbo", vmin=0, vmax=1)
            axes2[i].set_title(f"{k.replace('_',' ').title()} (0..{LBP_BINS-1})")
        else:
            disp = norm01(m)
            im = axes2[i].imshow(disp, cmap="turbo", vmin=0, vmax=1)
            axes2[i].set_title(k.replace('_',' ').title())
        axes2[i].set_xticks([]); axes2[i].set_yticks([])
        plt.colorbar(im, ax=axes2[i], fraction=0.046, pad=0.04)
    fig2.suptitle("LBP Feature Heatmaps (8×8 grid)", fontsize=13)
    fig2.tight_layout(rect=[0,0,1,0.95])
    plt.show()

    # ---- FIGURE 3: Global LBP histogram (+ top bins) ----
    hist = lbp_hist(lbp.ravel(), bins=LBP_BINS)
    xs = np.arange(LBP_BINS)
    topk = min(5, LBP_BINS)
    peaks = xs[np.argsort(hist)[::-1][:topk]]

    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 4), dpi=110)
    ax3.bar(xs, hist, color="steelblue", width=0.9)
    for p in peaks:
        ax3.axvline(p, color="orange", lw=1.2, alpha=0.9)
    ax3.set_title(f"Global LBP Histogram (uniform, bins={LBP_BINS}) — top bins: {peaks.tolist()}")
    ax3.set_xlabel("LBP code"); ax3.set_ylabel("Count")
    plt.tight_layout()
    plt.show()

    # ---- Save outputs ----
    base = os.path.join(OUT_DIR, save_prefix)
    cv2.imwrite(base + "_original.jpg", img_bgr)
    cv2.imwrite(base + "_gray.jpg", gray)
    plt.imsave(base + "_lbp_map.png", lbp_norm, cmap="turbo")
    cv2.imwrite(base + "_lbp_overlay.jpg", overlay)

    # save heatmaps as images
    ent_disp = norm01(maps["entropy"])
    dom_disp = maps["dominant_bin"] / (LBP_BINS - 1)
    plt.imsave(base + "_entropy_heatmap.png", ent_disp, cmap="turbo")
    plt.imsave(base + "_dominantbin_heatmap.png", dom_disp, cmap="turbo")

    # save histogram plot
    fig3_save, ax3s = plt.subplots(1, 1, figsize=(10, 4), dpi=110)
    ax3s.bar(xs, hist, color="steelblue", width=0.9)
    for p in peaks:
        ax3s.axvline(p, color="orange", lw=1.2, alpha=0.9)
    ax3s.set_title(f"Global LBP Histogram (uniform) — top bins: {peaks.tolist()}")
    ax3s.set_xlabel("LBP code"); ax3s.set_ylabel("Count")
    fig3_save.tight_layout()
    fig3_save.savefig(base + "_hist.png", bbox_inches="tight")
    plt.close(fig3_save)

    print(f"💾 Saved under: {OUT_DIR} with prefix {os.path.basename(base)}")

# ----------- RUN (single image) -----------
if __name__ == "__main__":
    demo_img = "26102010062.jpg"  # change to any filename in IMG_DIR
    p = os.path.join(IMG_DIR, demo_img)
    if not os.path.exists(p):
        imgs = [f for f in sorted(os.listdir(IMG_DIR)) if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
        if not imgs:
            raise SystemExit(f"No images found in {IMG_DIR}")
        p = os.path.join(IMG_DIR, imgs[0])

    print("Using image:", p)
    show_and_save_lbp_demo(p, save_prefix="lbp_demo_01")
