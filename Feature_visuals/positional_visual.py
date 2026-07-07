#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Positional features visual demonstration on 8×8 grid:
- Original
- X-position overlay (left→right)
- Y-position overlay (top→bottom)
- Center-bias overlay (1 - radial distance from center)
- 8×8 heatmaps for X, Y, Center-bias
- Saves images + a CSV of the 8×8 numbers
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------- PATHS -----------
BASE    = "/home/pratyush/Desktop/DS_Project"
IMG_DIR = f"{BASE}/data/preprocessed"       # your 800×600 images
OUT_DIR = f"{BASE}/outputs/pos_demo"
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

def build_positional_maps(gw=GW, gh=GH):
    """
    Returns 8×8 maps:
      X_map[r,c] = (c + 0.5)/gw   -> 0 (left) .. 1 (right)
      Y_map[r,c] = (r + 0.5)/gh   -> 0 (top)  .. 1 (bottom)
      C_map      = 1 - radial distance to center in [0,1] (center-bias)
    """
    c_idx = (np.arange(gw) + 0.5) / gw  # shape (gw,)
    r_idx = (np.arange(gh) + 0.5) / gh  # shape (gh,)
    X_map = np.tile(c_idx, (gh, 1)).astype(np.float32)
    Y_map = np.tile(r_idx[:, None], (1, gw)).astype(np.float32)

    # radial distance from center (0.5, 0.5)
    dx = X_map - 0.5
    dy = Y_map - 0.5
    radial = np.sqrt(dx*dx + dy*dy)
    radial /= np.sqrt((0.5**2) + (0.5**2))  # normalize to [0,1] w.r.t. farthest corner
    C_map = 1.0 - radial  # center-bias: 1 at center, 0 near corners
    return X_map, Y_map, C_map

def upsample_to_image(map_2d, W=W, H=H):
    """
    Upsample GH×GW map to H×W for pretty overlays.
    Assumes map_2d is already in [0,1].
    """
    m_img = cv2.resize(map_2d.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    return m_img

def overlay_heatmap(img_bgr, norm_map, alpha_img=0.6, alpha_heat=0.6, cmap_name="turbo"):
    """
    norm_map: H×W in [0,1]
    """
    cmap = plt.get_cmap(cmap_name)
    heat_rgb = (cmap(norm_map)[..., :3] * 255).astype(np.uint8)[:, :, ::-1]  # RGB→BGR
    return cv2.addWeighted(img_bgr, alpha_img, heat_rgb, alpha_heat, 0)

def save_heatmap_png(path, map_2d, cmap="turbo"):
    plt.imsave(path, map_2d, cmap=cmap, vmin=0, vmax=1)

# ----------- MAIN DEMO -----------
def show_and_save_pos_demo(image_path, save_prefix="pos_demo_01"):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"⚠️ Cannot read: {image_path}")
        return

    img_bgr = ensure_size(img_bgr, W, H)

    # Build positional 8×8 maps
    X_map, Y_map, C_map = build_positional_maps(GW, GH)

    # Upsample to image size for overlays
    X_img = upsample_to_image(X_map, W, H)  # 0..1 left→right
    Y_img = upsample_to_image(Y_map, W, H)  # 0..1 top→bottom
    C_img = upsample_to_image(C_map, W, H)  # 0..1 center-bias

    # Overlays
    x_overlay = overlay_heatmap(img_bgr, X_img, alpha_img=0.65, alpha_heat=0.55, cmap_name="turbo")
    y_overlay = overlay_heatmap(img_bgr, Y_img, alpha_img=0.65, alpha_heat=0.55, cmap_name="turbo")
    c_overlay = overlay_heatmap(img_bgr, C_img, alpha_img=0.65, alpha_heat=0.55, cmap_name="turbo")

    # ---- FIGURE 1: Original + X overlay + Y overlay + Center overlay ----
    fig1, axes1 = plt.subplots(1, 4, figsize=(18, 4.8), dpi=110)
    axes1[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); axes1[0].set_title("Original"); axes1[0].axis("off")

    im1 = axes1[1].imshow(X_img, cmap="turbo", vmin=0, vmax=1); axes1[1].set_title("X position (0→1)"); axes1[1].axis("off")
    plt.colorbar(im1, ax=axes1[1], fraction=0.046, pad=0.04)

    im2 = axes1[2].imshow(Y_img, cmap="turbo", vmin=0, vmax=1); axes1[2].set_title("Y position (0→1)"); axes1[2].axis("off")
    plt.colorbar(im2, ax=axes1[2], fraction=0.046, pad=0.04)

    im3 = axes1[3].imshow(C_img, cmap="turbo", vmin=0, vmax=1); axes1[3].set_title("Center-bias (1=center)"); axes1[3].axis("off")
    plt.colorbar(im3, ax=axes1[3], fraction=0.046, pad=0.04)

    fig1.tight_layout()
    plt.show()

    # ---- FIGURE 2: Overlays side-by-side ----
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.8), dpi=110)
    axes2[0].imshow(cv2.cvtColor(x_overlay, cv2.COLOR_BGR2RGB)); axes2[0].set_title("X-position overlay"); axes2[0].axis("off")
    axes2[1].imshow(cv2.cvtColor(y_overlay, cv2.COLOR_BGR2RGB)); axes2[1].set_title("Y-position overlay"); axes2[1].axis("off")
    axes2[2].imshow(cv2.cvtColor(c_overlay, cv2.COLOR_BGR2RGB)); axes2[2].set_title("Center-bias overlay"); axes2[2].axis("off")
    fig2.tight_layout()
    plt.show()

    # ---- FIGURE 3: 8×8 heatmaps (X, Y, Center-bias) ----
    fig3, axes3 = plt.subplots(1, 3, figsize=(10.5, 4.5), dpi=110)
    for ax, m, title in zip(axes3,
                            [X_map, Y_map, C_map],
                            ["X map (8×8)", "Y map (8×8)", "Center-bias (8×8)"]):
        im = ax.imshow(m, cmap="turbo", vmin=0, vmax=1)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig3.suptitle("Positional Feature Heatmaps", fontsize=13)
    fig3.tight_layout(rect=[0,0,1,0.95])
    plt.show()

    # ---- Save outputs ----
    base = os.path.join(OUT_DIR, save_prefix)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Save images
    cv2.imwrite(base + "_original.jpg", img_bgr)
    save_heatmap_png(base + "_x_heatmap.png", X_img, cmap="turbo")
    save_heatmap_png(base + "_y_heatmap.png", Y_img, cmap="turbo")
    save_heatmap_png(base + "_center_heatmap.png", C_img, cmap="turbo")
    cv2.imwrite(base + "_x_overlay.jpg", x_overlay)
    cv2.imwrite(base + "_y_overlay.jpg", y_overlay)
    cv2.imwrite(base + "_center_overlay.jpg", c_overlay)

    # Save the raw 8×8 numbers to CSV (nice for reports)
    df = pd.DataFrame({
        "cell": [f"c{i:02d}" for i in range(1, GW*GH+1)],
        "x": X_map.reshape(-1),
        "y": Y_map.reshape(-1),
        "center_bias": C_map.reshape(-1)
    })
    df.to_csv(base + "_positional_8x8.csv", index=False)

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
    show_and_save_pos_demo(p, save_prefix="pos_demo_01")
