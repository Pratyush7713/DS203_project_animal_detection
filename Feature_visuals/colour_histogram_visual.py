#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colour Histogram visual demonstration (RGB & HSV):
- Original → RGB histograms → Hue histogram (+ swatches) → Hue map → Dominant hue overlay
- Saves results under outputs/color_hist_demo/
"""

import os
import warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Unable to import Axes3D")

# ----------- PATHS -----------
BASE    = "/home/pratyush/Desktop/DS_Project"
IMG_DIR = f"{BASE}/data/preprocessed"        # your 800x600 images
OUT_DIR = f"{BASE}/outputs/color_hist_demo"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------- HELPERS -----------
def ensure_size(img_bgr, w=800, h=600):
    if img_bgr.shape[1] != w or img_bgr.shape[0] != h:
        img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    return img_bgr

def rgb_histograms(img_bgr, bins=256):
    """Compute R, G, B histograms (0..255). Returns arrays of shape (bins,)."""
    b, g, r = cv2.split(img_bgr)
    histB = cv2.calcHist([b],[0],None,[bins],[0,256]).flatten()
    histG = cv2.calcHist([g],[0],None,[bins],[0,256]).flatten()
    histR = cv2.calcHist([r],[0],None,[bins],[0,256]).flatten()
    return histR, histG, histB

def hsv_hists(img_bgr, hue_bins=180, sat_bins=256):
    """Hue histogram (0..179), and 2D Hue-Saturation histogram."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hue_hist = cv2.calcHist([h],[0],None,[hue_bins],[0,180]).flatten()
    # HS 2D histogram (for optional visualization/analysis)
    hs_hist = cv2.calcHist([h, s],[0,1],None,[hue_bins, sat_bins],[0,180, 0,256])
    return hue_hist, hs_hist, hsv

def hue_map_visual(img_bgr, s_thresh=50, v_thresh=50):
    """
    Visualize hue per pixel:
    - keep hue color where saturation & value are decent
    - gray out low-saturation/value regions
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (s >= s_thresh) & (v >= v_thresh)
    # make a vivid HSV: keep H, set S=255, V=200 for strong color
    hsv_vis = hsv.copy()
    hsv_vis[..., 1] = 255
    hsv_vis[..., 2] = 200
    hue_img = cv2.cvtColor(hsv_vis, cv2.COLOR_HSV2BGR)
    # gray fallback
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out = np.where(mask[..., None], hue_img, gray3)
    return out, mask

def dominant_hue_overlay(img_bgr, hsv, top_k=1, tol_bins=5, s_thresh=50, v_thresh=50):
    """
    Highlight pixels belonging to the top-K dominant hue bins (±tol_bins) with sufficient S & V.
    Returns overlay image and the list of dominant hue centers (in degrees 0..179).
    """
    h, s, v = cv2.split(hsv)
    # consider only reasonably colorful & bright pixels
    valid = (s >= s_thresh) & (v >= v_thresh)
    h_valid = h[valid]
    if h_valid.size == 0:
        # nothing colorful; return original
        return img_bgr.copy(), []

    # hue hist on valid pixels
    hue_bins = 180
    hist, _ = np.histogram(h_valid, bins=hue_bins, range=(0,180))
    # pick top-K peaks
    top_idx = np.argsort(hist)[::-1][:top_k]
    dom_hues = top_idx.tolist()  # 0..179 bin indices

    # build mask for any pixel whose hue falls within ±tol_bins of any dominant hue
    mask = np.zeros_like(h, dtype=bool)
    for dh in dom_hues:
        low = (dh - tol_bins) % 180
        high = (dh + tol_bins) % 180
        if low <= high:
            band = (h >= low) & (h <= high)
        else:
            # wrap-around case
            band = (h >= low) | (h <= high)
        mask |= (band & valid)

    # overlay mask on original
    overlay = img_bgr.copy()
    color = (0, 255, 255)  # yellow highlight
    alpha = 0.45
    highlight = np.zeros_like(overlay, dtype=np.uint8)
    highlight[:] = color
    overlay = np.where(mask[..., None], cv2.addWeighted(overlay, 1-alpha, highlight, alpha, 0), overlay)

    return overlay, dom_hues

def draw_rgb_hist_panel(ax, histR, histG, histB, bins=256):
    xs = np.arange(bins)
    ax.plot(xs, histR, color='r', lw=1, label='R')
    ax.plot(xs, histG, color='g', lw=1, label='G')
    ax.plot(xs, histB, color='b', lw=1, label='B')
    ax.set_xlim(0, bins-1)
    ax.set_title("RGB Histograms (256 bins)")
    ax.set_xlabel("Intensity"); ax.set_ylabel("Count")
    ax.legend(loc="upper right", fontsize=8)

def draw_hue_hist_and_swatches(ax_hist, ax_swatch, hue_hist, top_k=5):
    xs = np.arange(180)
    ax_hist.bar(xs, hue_hist, color='gray', width=1.0)
    ax_hist.set_xlim(0,179)
    ax_hist.set_title("Hue Histogram (0..179)")
    ax_hist.set_xlabel("Hue bin"); ax_hist.set_ylabel("Count")

    # Mark top-K hues
    peaks = np.argsort(hue_hist)[::-1][:top_k]
    for p in peaks:
        ax_hist.axvline(p, color='orange', lw=1.2, alpha=0.9)

    # Make swatches for top-K hues
    swatch_h = 40
    swatch = np.zeros((swatch_h*top_k, 200, 3), dtype=np.uint8)
    for i, hbin in enumerate(peaks):
        hsv_color = np.array([[[hbin, 255, 220]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0,0,:]
        swatch[i*swatch_h:(i+1)*swatch_h, :, :] = bgr
    ax_swatch.imshow(cv2.cvtColor(swatch, cv2.COLOR_BGR2RGB))
    ax_swatch.set_title("Top hue swatches")
    ax_swatch.axis("off")

# ----------- MAIN DEMO -----------
def show_and_save_color_demo(image_path, save_prefix="color_demo_01", top_k_hues=5):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"⚠️ Cannot read: {image_path}")
        return
    img_bgr = ensure_size(img_bgr, 800, 600)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # RGB hists
    histR, histG, histB = rgb_histograms(img_bgr, bins=256)

    # HSV hists & hue map
    hue_hist, hs_hist, hsv = hsv_hists(img_bgr, hue_bins=180, sat_bins=256)
    hue_map_img, hv_mask = hue_map_visual(img_bgr, s_thresh=50, v_thresh=50)

    # Dominant hue overlay (top-1 for overlay; hist panel shows top-5)
    dom_overlay, dom_hues = dominant_hue_overlay(img_bgr, hsv, top_k=1, tol_bins=5, s_thresh=50, v_thresh=50)
    dom_hues_str = ", ".join(str(h) for h in dom_hues)

    # ---- FIGURE 1: Original + Hue Map + Dominant Hue Overlay ----
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4.8), dpi=110)
    axes1[0].imshow(img_rgb); axes1[0].set_title("Original"); axes1[0].axis("off")
    axes1[1].imshow(cv2.cvtColor(hue_map_img, cv2.COLOR_BGR2RGB)); axes1[1].set_title("Hue Map (low S/V grayed)"); axes1[1].axis("off")
    axes1[2].imshow(cv2.cvtColor(dom_overlay, cv2.COLOR_BGR2RGB))
    axes1[2].set_title(f"Dominant Hue Overlay (h={dom_hues_str if dom_hues else 'N/A'})")
    axes1[2].axis("off")
    fig1.tight_layout()
    plt.show()

    # ---- FIGURE 2: RGB hist + Hue hist + Top hue swatches ----
    fig2 = plt.figure(figsize=(14, 5), dpi=110)
    gs = fig2.add_gridspec(2, 3, height_ratios=[2.2, 1.0])
    ax_rgb  = fig2.add_subplot(gs[:, 0])   # tall left (RGB hists)
    ax_hue  = fig2.add_subplot(gs[0, 1:]) # top right (Hue hist)
    ax_swat = fig2.add_subplot(gs[1, 1:]) # bottom right (swatches)

    draw_rgb_hist_panel(ax_rgb, histR, histG, histB, bins=256)
    draw_hue_hist_and_swatches(ax_hue, ax_swat, hue_hist, top_k=top_k_hues)
    fig2.suptitle("Colour Histogram Demonstration", fontsize=13)
    fig2.tight_layout(rect=[0,0,1,0.95])
    plt.show()

    # ---- Save outputs ----
    base = os.path.join(OUT_DIR, save_prefix)
    os.makedirs(OUT_DIR, exist_ok=True)
    # images
    cv2.imwrite(base + "_original.jpg", img_bgr)
    cv2.imwrite(base + "_hue_map.jpg", hue_map_img)
    cv2.imwrite(base + "_dominant_hue_overlay.jpg", dom_overlay)
    # plots
    # re-draw plots for saving quietly:
    # RGB + Hue + Swatches panel
    fig2_save = plt.figure(figsize=(14, 5), dpi=110)
    gs2 = fig2_save.add_gridspec(2, 3, height_ratios=[2.2, 1.0])
    ax_rgb2  = fig2_save.add_subplot(gs2[:, 0])
    ax_hue2  = fig2_save.add_subplot(gs2[0, 1:])
    ax_swat2 = fig2_save.add_subplot(gs2[1, 1:])
    draw_rgb_hist_panel(ax_rgb2, histR, histG, histB, bins=256)
    draw_hue_hist_and_swatches(ax_hue2, ax_swat2, hue_hist, top_k=top_k_hues)
    fig2_save.tight_layout()
    fig2_save.savefig(base + "_plots.png", bbox_inches="tight")
    plt.close(fig2_save)

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
    show_and_save_color_demo(p, save_prefix="color_demo_01", top_k_hues=5)
