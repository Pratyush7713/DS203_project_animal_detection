#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global Colour Moments (GCM) visual demo + 8×8 per-cell maps + reconstructions
- Global moments: mean, std, skew for RGB & HSV (bar charts)
- 8×8 heatmaps for HSV mean/std/skew
- Cell-wise mean reconstructions (RGB & HSV) to show "effect" of GCM
- Std/Skew overlays on the image
- Saves plots + CSV of the 8×8 HSV moments
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------- PATHS -----------
BASE    = "/home/pratyush/Desktop/DS_Project"
IMG_DIR = f"{BASE}/data/preprocessed"       # your 800×600 images
OUT_DIR = f"{BASE}/outputs/gcm_demo"
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

def moments_1d(x):
    """
    Return mean, std, skew for a 1D float array.
    Skew is the third standardized moment: E[((x - μ)/σ)^3].
    """
    x = x.astype(np.float64).ravel()
    mu = np.mean(x)
    sigma = np.std(x) + 1e-8
    skew = np.mean(((x - mu) / sigma) ** 3)
    # return (mean, std, skew) — std via RMS is equivalent to np.std
    return mu, np.sqrt(np.mean((x - mu)**2)), skew

def global_colour_moments(img_bgr):
    """
    Compute GCM for RGB and HSV channels.
    Returns dicts: {'R':(μ,σ,γ), 'G':..., 'B':...} and {'H':...,'S':...,'V':...}
    """
    b, g, r = cv2.split(img_bgr)
    rgb = {'R': moments_1d(r), 'G': moments_1d(g), 'B': moments_1d(b)}
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)  # H:0..179, S:0..255, V:0..255
    hsv_m = {'H': moments_1d(h), 'S': moments_1d(s), 'V': moments_1d(v)}
    return rgb, hsv_m

def per_cell_hsv_moments(img_bgr):
    """
    Compute per-cell (8×8) HSV moments: mean/std/skew for H,S,V.
    Returns dict of maps: {('H','mean'): GH×GW, ('H','std'):..., ('H','skew'):...}
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    Hc, Sc, Vc = cv2.split(hsv)
    out = {}
    for ch_name, ch in [('H', Hc), ('S', Sc), ('V', Vc)]:
        mean_map = np.zeros((GH, GW), np.float32)
        std_map  = np.zeros((GH, GW), np.float32)
        skew_map = np.zeros((GH, GW), np.float32)
        for r in range(GH):
            for c in range(GW):
                y0, y1 = r*CH, (r+1)*CH
                x0, x1 = c*CW, (c+1)*CW
                block = ch[y0:y1, x0:x1]
                mu, sd, sk = moments_1d(block)
                mean_map[r, c] = mu
                std_map[r, c]  = sd
                skew_map[r, c] = sk
        out[(ch_name,'mean')] = mean_map
        out[(ch_name,'std')]  = std_map
        out[(ch_name,'skew')] = skew_map
    return out

def norm01(x):
    x = x.astype(np.float32)
    return (x - np.min(x)) / (np.ptp(x) + 1e-8)

def save_heatmap(ax, data2d, title):
    disp = norm01(data2d)
    im = ax.imshow(disp, cmap="turbo", vmin=0, vmax=1)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im, cb

def barplot_gcm(ax, gcm_dict, space_title):
    """
    gcm_dict: {'R':(μ,σ,γ), 'G':..., ...}
    Plots 3 bars per channel for that color space.
    """
    chans = list(gcm_dict.keys())
    means = [gcm_dict[ch][0] for ch in chans]
    stds  = [gcm_dict[ch][1] for ch in chans]
    skews = [gcm_dict[ch][2] for ch in chans]

    x = np.arange(len(chans))
    width = 0.25
    ax.bar(x - width, means, width, label='mean')
    ax.bar(x,         stds,  width, label='std')
    ax.bar(x + width, skews, width, label='skew')
    ax.set_xticks(x); ax.set_xticklabels(chans)
    ax.set_title(f"{space_title} — Global Colour Moments")
    ax.legend(fontsize=8)

# ======= NEW: "Effect" of GCM via cell-wise mean reconstructions & overlays =======
def reconstruct_cellwise_mean(img_bgr, space="HSV"):
    """
    Reconstruct image by replacing each 8×8 cell with its mean colour
    (in RGB or HSV). This visually shows the 'effect' of GCM.
    """
    out = np.zeros_like(img_bgr)
    if space.upper() == "HSV":
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        for r in range(GH):
            for c in range(GW):
                y0, y1 = r*CH, (r+1)*CH
                x0, x1 = c*CW, (c+1)*CW
                mH = np.mean(H[y0:y1, x0:x1])
                mS = np.mean(S[y0:y1, x0:x1])
                mV = np.mean(V[y0:y1, x0:x1])
                block = np.zeros((CH, CW, 3), dtype=np.uint8)
                block[..., 0] = np.uint8(np.clip(mH, 0, 179))
                block[..., 1] = np.uint8(np.clip(mS, 0, 255))
                block[..., 2] = np.uint8(np.clip(mV, 0, 255))
                out[y0:y1, x0:x1] = cv2.cvtColor(block, cv2.COLOR_HSV2BGR)
    else:  # RGB
        B, G, R = cv2.split(img_bgr)
        for r in range(GH):
            for c in range(GW):
                y0, y1 = r*CH, (r+1)*CH
                x0, x1 = c*CW, (c+1)*CW
                mB = np.mean(B[y0:y1, x0:x1])
                mG = np.mean(G[y0:y1, x0:x1])
                mR = np.mean(R[y0:y1, x0:x1])
                out[y0:y1, x0:x1, 0] = np.uint8(np.clip(mB, 0, 255))
                out[y0:y1, x0:x1, 1] = np.uint8(np.clip(mG, 0, 255))
                out[y0:y1, x0:x1, 2] = np.uint8(np.clip(mR, 0, 255))
    return out

def upsample(map_2d):
    disp = (map_2d - np.min(map_2d)) / (np.ptp(map_2d) + 1e-8)
    return cv2.resize(disp.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)

def overlay_heat(img_bgr, norm_map, cmap_name="turbo",
                 alpha_img=0.65, alpha_heat=0.55):
    cmap = plt.get_cmap(cmap_name)
    heat_rgb = (cmap(norm_map)[..., :3] * 255).astype(np.uint8)[:, :, ::-1]
    return cv2.addWeighted(img_bgr, alpha_img, heat_rgb, alpha_heat, 0)

# ----------- MAIN DEMO -----------
def show_and_save_gcm_demo(image_path, save_prefix="gcm_demo_01"):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"⚠️ Cannot read: {image_path}")
        return

    img_bgr = ensure_size(img_bgr, W, H)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ===== 1) Global moments (RGB & HSV) =====
    gcm_rgb, gcm_hsv = global_colour_moments(img_bgr)

    fig1, axes1 = plt.subplots(1, 3, figsize=(16, 4.8), dpi=110)
    axes1[0].imshow(img_rgb); axes1[0].set_title("Original"); axes1[0].axis("off")
    barplot_gcm(axes1[1], gcm_rgb, "RGB")
    barplot_gcm(axes1[2], gcm_hsv, "HSV")
    fig1.tight_layout()
    plt.show()

    # ===== 2) 8×8 per-cell HSV moments =====
    maps = per_cell_hsv_moments(img_bgr)

    # A) Mean heatmaps for H, S, V
    fig2, ax2 = plt.subplots(1, 3, figsize=(14, 4.5), dpi=110)
    save_heatmap(ax2[0], maps[('H','mean')], "H mean (8×8)")
    save_heatmap(ax2[1], maps[('S','mean')], "S mean (8×8)")
    save_heatmap(ax2[2], maps[('V','mean')], "V mean (8×8)")
    fig2.suptitle("HSV Mean Heatmaps", fontsize=13)
    fig2.tight_layout(rect=[0,0,1,0.95])
    plt.show()

    # B) Std heatmaps
    fig3, ax3 = plt.subplots(1, 3, figsize=(14, 4.5), dpi=110)
    save_heatmap(ax3[0], maps[('H','std')], "H std (8×8)")
    save_heatmap(ax3[1], maps[('S','std')], "S std (8×8)")
    save_heatmap(ax3[2], maps[('V','std')], "V std (8×8)")
    fig3.suptitle("HSV Std Heatmaps", fontsize=13)
    fig3.tight_layout(rect=[0,0,1,0.95])
    plt.show()

    # C) Skew heatmaps
    fig4, ax4 = plt.subplots(1, 3, figsize=(14, 4.5), dpi=110)
    save_heatmap(ax4[0], maps[('H','skew')], "H skew (8×8)")
    save_heatmap(ax4[1], maps[('S','skew')], "S skew (8×8)")
    save_heatmap(ax4[2], maps[('V','skew')], "V skew (8×8)")
    fig4.suptitle("HSV Skew Heatmaps", fontsize=13)
    fig4.tight_layout(rect=[0,0,1,0.95])
    plt.show()

    # ===== 3) Show the 'effect' of GCM via cell-wise mean reconstructions =====
    recon_hsv = reconstruct_cellwise_mean(img_bgr, space="HSV")
    recon_rgb = reconstruct_cellwise_mean(img_bgr, space="RGB")

    figR, axR = plt.subplots(1, 3, figsize=(16, 4.8), dpi=110)
    axR[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); axR[0].set_title("Original"); axR[0].axis("off")
    axR[1].imshow(cv2.cvtColor(recon_rgb, cv2.COLOR_BGR2RGB)); axR[1].set_title("GCM reconstruction (RGB mean per cell)"); axR[1].axis("off")
    axR[2].imshow(cv2.cvtColor(recon_hsv, cv2.COLOR_BGR2RGB)); axR[2].set_title("GCM reconstruction (HSV mean per cell)"); axR[2].axis("off")
    figR.tight_layout()
    plt.show()

    # ===== 4) Variability & asymmetry overlays (std/skew) =====
    V_std_img  = upsample(maps[('V','std')])   # brightness variability
    S_std_img  = upsample(maps[('S','std')])   # saturation variability
    H_skew_img = upsample(maps[('H','skew')])  # hue asymmetry

    ov1 = overlay_heat(img_bgr, V_std_img)
    ov2 = overlay_heat(img_bgr, S_std_img)
    ov3 = overlay_heat(img_bgr, H_skew_img)

    figO, axO = plt.subplots(1, 3, figsize=(16, 4.8), dpi=110)
    axO[0].imshow(cv2.cvtColor(ov1, cv2.COLOR_BGR2RGB)); axO[0].set_title("Overlay: Value (V) std"); axO[0].axis("off")
    axO[1].imshow(cv2.cvtColor(ov2, cv2.COLOR_BGR2RGB)); axO[1].set_title("Overlay: Saturation (S) std"); axO[1].axis("off")
    axO[2].imshow(cv2.cvtColor(ov3, cv2.COLOR_BGR2RGB)); axO[2].set_title("Overlay: Hue (H) skew"); axO[2].axis("off")
    figO.tight_layout()
    plt.show()

    # ===== Save outputs =====
    base = os.path.join(OUT_DIR, save_prefix)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Save global plots (re-draw quickly to file)
    fig1_save, axs1s = plt.subplots(1, 3, figsize=(16, 4.8), dpi=110)
    axs1s[0].imshow(img_rgb); axs1s[0].set_title("Original"); axs1s[0].axis("off")
    barplot_gcm(axs1s[1], gcm_rgb, "RGB")
    barplot_gcm(axs1s[2], gcm_hsv, "HSV")
    fig1_save.tight_layout()
    fig1_save.savefig(base + "_global_gcm.png", bbox_inches="tight")
    plt.close(fig1_save)

    # 2) Save heatmaps as PNGs
    for (ch, stat) in [('H','mean'),('S','mean'),('V','mean'),
                       ('H','std'),('S','std'),('V','std'),
                       ('H','skew'),('S','skew'),('V','skew')]:
        plt.imsave(base + f"_{ch}_{stat}_heatmap.png", norm01(maps[(ch,stat)]), cmap="turbo", vmin=0, vmax=1)

    # 3) Save raw 8×8 numbers to CSV
    rows = []
    for r in range(GH):
        for c in range(GW):
            rows.append({
                "cell": f"c{r*GW + c + 1:02d}",
                "H_mean": maps[('H','mean')][r,c],
                "S_mean": maps[('S','mean')][r,c],
                "V_mean": maps[('V','mean')][r,c],
                "H_std":  maps[('H','std')][r,c],
                "S_std":  maps[('S','std')][r,c],
                "V_std":  maps[('V','std')][r,c],
                "H_skew": maps[('H','skew')][r,c],
                "S_skew": maps[('S','skew')][r,c],
                "V_skew": maps[('V','skew')][r,c],
            })
    pd.DataFrame(rows).to_csv(base + "_hsv_moments_8x8.csv", index=False)

    # 4) Save reconstructions & overlays
    cv2.imwrite(base + "_recon_rgb_mean.jpg", recon_rgb)
    cv2.imwrite(base + "_recon_hsv_mean.jpg", recon_hsv)
    cv2.imwrite(base + "_overlay_V_std.jpg", ov1)
    cv2.imwrite(base + "_overlay_S_std.jpg", ov2)
    cv2.imwrite(base + "_overlay_H_skew.jpg", ov3)

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
    show_and_save_gcm_demo(p, save_prefix="gcm_demo_01")