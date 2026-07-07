#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLCM visual demonstration on 8×8 grid:
- Original → Grayscale
- 8×8 heatmaps for Contrast, Homogeneity, Energy(ASM), Correlation
- Contrast overlay (upsampled) on the image
Saves outputs under outputs/glcm_demo/
"""

import os
import warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt

# skimage 0.25+ location for GLCM:
from skimage.feature.texture import graycomatrix, graycoprops

warnings.filterwarnings("ignore", message="Unable to import Axes3D")

# ----------- PATHS -----------
BASE    = "/home/pratyush/Desktop/DS_Project"
IMG_DIR = f"{BASE}/data/preprocessed"       # your 800×600 set
OUT_DIR = f"{BASE}/outputs/glcm_demo"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------- GRID / GEOMETRY -----------
W, H = 800, 600
GW, GH = 8, 8
CW, CH = W // GW, H // GH  # 100×75

# ----------- HELPERS -----------
def ensure_size(img_bgr, w=W, h=H):
    if img_bgr.shape[1] != w or img_bgr.shape[0] != h:
        img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    return img_bgr

def to_gray_uint8(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray

def quantize_gray(gray_uint8, levels=32):
    """
    Map 0..255 → 0..levels-1 (int).
    Lower 'levels' = faster & more stable GLCM.
    """
    assert gray_uint8.dtype == np.uint8
    bins = np.linspace(0, 256, levels+1, endpoint=True)
    q = np.digitize(gray_uint8, bins) - 1
    q[q == levels] = levels - 1
    return q.astype(np.uint8)

def glcm_props_for_block(block_u8, levels=32, distances=(1, 2), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    """
    Compute GLCM and return averaged properties for a single block.
    Returns dict with keys: contrast, homogeneity, energy (ASM), correlation, dissimilarity
    """
    # block must be quantized levels: 0..levels-1 (uint8)
    glcm = graycomatrix(block_u8,
                        distances=distances,
                        angles=angles,
                        levels=levels,
                        symmetric=True,
                        normed=True)

    props = {}
    # graycoprops names: 'contrast','dissimilarity','homogeneity','ASM','energy','correlation'
    props['contrast']     = float(np.mean(graycoprops(glcm, 'contrast')))
    props['homogeneity']  = float(np.mean(graycoprops(glcm, 'homogeneity')))
    props['energy']       = float(np.mean(graycoprops(glcm, 'ASM')))   # ASM (energy^2); consistent naming
    props['correlation']  = float(np.mean(graycoprops(glcm, 'correlation')))
    props['dissimilarity']= float(np.mean(graycoprops(glcm, 'dissimilarity')))
    return props

def compute_glcm_grid_maps(gray_u8, levels=32):
    """
    Compute GLCM props per 8×8 cell, return 2D maps (GH×GW) for each property.
    """
    qimg = quantize_gray(gray_u8, levels=levels)
    maps = {
        'contrast': np.zeros((GH, GW), dtype=np.float32),
        'homogeneity': np.zeros((GH, GW), dtype=np.float32),
        'energy': np.zeros((GH, GW), dtype=np.float32),
        'correlation': np.zeros((GH, GW), dtype=np.float32),
        'dissimilarity': np.zeros((GH, GW), dtype=np.float32),
    }

    for r in range(GH):
        for c in range(GW):
            y0, y1 = r*CH, (r+1)*CH
            x0, x1 = c*CW, (c+1)*CW
            block = qimg[y0:y1, x0:x1]
            props = glcm_props_for_block(block, levels=levels)
            for k in maps.keys():
                maps[k][r, c] = props[k]
    return maps

def upsample_to_image(map_2d):
    """
    Upsample GH×GW map to 600×800 for pretty overlays.
    """
    # map_2d is (GH, GW) → resize to (H, W)
    m = map_2d.astype(np.float32)
    # normalize to 0..1 for color mapping
    m = (m - np.min(m)) / (np.ptp(m) + 1e-8)
    m_img = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return m_img

def overlay_heatmap(img_bgr, norm_map, alpha_img=0.6, alpha_heat=0.6, cmap_name="turbo"):
    cmap = plt.get_cmap(cmap_name)
    heat_rgb = (cmap(norm_map)[..., :3] * 255).astype(np.uint8)[:, :, ::-1]  # Matplotlib RGB → BGR
    overlay = cv2.addWeighted(img_bgr, alpha_img, heat_rgb, alpha_heat, 0)
    return overlay

# ----------- MAIN DEMO -----------
def show_and_save_glcm_demo(image_path, save_prefix="glcm_demo_01", levels=32):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"⚠️ Cannot read: {image_path}")
        return

    img_bgr = ensure_size(img_bgr, W, H)
    gray_u8 = to_gray_uint8(img_bgr)

    # Compute GLCM per grid cell
    maps = compute_glcm_grid_maps(gray_u8, levels=levels)

    # Pick four to display as heatmaps
    show_keys = ["contrast", "homogeneity", "energy", "correlation"]

    # Figure 1: Original + Grayscale + Contrast overlay
    contrast_norm = upsample_to_image(maps["contrast"])
    contrast_overlay = overlay_heatmap(img_bgr, contrast_norm, alpha_img=0.65, alpha_heat=0.55, cmap_name="turbo")

    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4.8), dpi=110)
    axes1[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); axes1[0].set_title("Original"); axes1[0].axis("off")
    axes1[1].imshow(gray_u8, cmap='gray'); axes1[1].set_title("Grayscale"); axes1[1].axis("off")
    im = axes1[2].imshow(contrast_norm, cmap="turbo", vmin=0, vmax=1)
    axes1[2].set_title("GLCM Contrast (8×8 map)")
    axes1[2].set_xticks([]); axes1[2].set_yticks([])
    plt.colorbar(im, ax=axes1[2], fraction=0.046, pad=0.04)
    fig1.tight_layout()
    plt.show()

    # Figure 2: 4 heatmaps side-by-side (GH×GW each)
    fig2, axes2 = plt.subplots(1, len(show_keys), figsize=(4.2*len(show_keys), 4.5), dpi=110)
    for i, k in enumerate(show_keys):
        m = maps[k]
        # normalize per-map for display
        disp = (m - np.min(m)) / (np.ptp(m) + 1e-8)
        im = axes2[i].imshow(disp, cmap="turbo", vmin=0, vmax=1)
        axes2[i].set_title(f"{k.capitalize()}"); axes2[i].set_xticks([]); axes2[i].set_yticks([])
        plt.colorbar(im, ax=axes2[i], fraction=0.046, pad=0.04)
    fig2.suptitle("GLCM Feature Heatmaps (8×8 grid)", fontsize=13)
    fig2.tight_layout(rect=[0,0,1,0.96])
    plt.show()

    # Figure 3: Contrast overlay upsampled to image size
    fig3, axes3 = plt.subplots(1, 2, figsize=(10.5, 4.8), dpi=110)
    axes3[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); axes3[0].set_title("Original"); axes3[0].axis("off")
    axes3[1].imshow(cv2.cvtColor(contrast_overlay, cv2.COLOR_BGR2RGB)); axes3[1].set_title("Contrast Overlay"); axes3[1].axis("off")
    fig3.tight_layout()
    plt.show()

    # ---- Save outputs ----
    base = os.path.join(OUT_DIR, save_prefix)
    cv2.imwrite(base + "_original.jpg", img_bgr)
    cv2.imwrite(base + "_gray.jpg", gray_u8)
    # Save raw heatmaps (as images) for each key
    for k in show_keys:
        disp = (maps[k] - np.min(maps[k])) / (np.ptp(maps[k]) + 1e-8)
        plt.imsave(base + f"_{k}_heatmap.png", disp, cmap="turbo")
    cv2.imwrite(base + "_contrast_overlay.jpg", contrast_overlay)
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
    # levels=32 keeps it fast & stable; bump to 64 if you want more texture detail (slower)
    show_and_save_glcm_demo(p, save_prefix="glcm_demo_01", levels=32)
