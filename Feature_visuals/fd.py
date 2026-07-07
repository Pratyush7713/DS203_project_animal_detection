#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fourier Descriptors (FD) visual demo with 8×8 grid overlays
- Extract dominant contour from image (Canny edges)
- Compute Fourier descriptors (translation/scale/rotation invariant)
- Progressive reconstructions with K descriptors (10/20/50/100)
- Descriptor magnitude spectrum (log)
- Overlay reconstructed contour on the original + 8×8 grid
- Edge-density 8×8 heatmap
- Save first 128 FD magnitudes to CSV
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------- paths --------
BASE    = "/home/pratyush/Desktop/DS_Project"
IMG_DIR = f"{BASE}/data/preprocessed"     # 800×600 images
OUT_DIR = f"{BASE}/outputs/fd_demo"
os.makedirs(OUT_DIR, exist_ok=True)

# -------- grid --------
W, H = 800, 600
GW, GH = 8, 8
CW, CH = W // GW, H // GH  # 100×75

# -------- utils --------
def ensure_size(img_bgr, w=W, h=H):
    if img_bgr.shape[1] != w or img_bgr.shape[0] != h:
        img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    return img_bgr

def draw_grid(ax):
    for c in range(1, GW):
        ax.plot([c*CW, c*CW], [0, H], color='yellow', lw=0.8, alpha=0.7)
    for r in range(1, GH):
        ax.plot([0, W], [r*CH, r*CH], color='yellow', lw=0.8, alpha=0.7)
    # cell numbers (bottom-right)
    idx = 1
    for r in range(GH):
        for c in range(GW):
            x = (c+1)*CW - 5
            y = (r+1)*CH - 5
            ax.text(x, y, f"{idx:02d}", color='yellow', fontsize=7,
                    va='top', ha='right')
            idx += 1

def auto_canny_thresholds(gray):
    # median heuristic
    v = np.median(gray)
    low = int(max(0, 0.66 * v))
    high = int(min(255, 1.33 * v))
    return low, high

def extract_dominant_contour(img_bgr):
    """
    Returns contour Nx2 (x,y) or None.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 1.0)

    low, high = auto_canny_thresholds(gray)
    edges = cv2.Canny(gray, low, high)

    # find contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, edges

    # select contour with max area; fall back to longest perimeter if needed
    areas = [cv2.contourArea(c) for c in cnts]
    max_idx = int(np.argmax(areas))
    dom = cnts[max_idx]
    if areas[max_idx] < 10:  # too tiny; choose longest
        perims = [cv2.arcLength(c, True) for c in cnts]
        dom = cnts[int(np.argmax(perims))]

    # reshape to Nx2 float
    dom = dom.reshape(-1, 2).astype(np.float64)
    return dom, edges

def resample_contour(contour, n=512):
    """
    Re-sample contour points to length n along cumulative arc-length
    for stable FD (uniform sampling).
    """
    if len(contour) < 5:
        return None
    # close contour
    pts = np.vstack([contour, contour[0]])
    d = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    s = np.concatenate([[0], np.cumsum(d)])
    total = s[-1]
    if total < 1e-3:
        return None
    s_new = np.linspace(0, total, n, endpoint=False)
    x = np.interp(s_new, s, pts[:,0])
    y = np.interp(s_new, s, pts[:,1])
    return np.stack([x, y], axis=1)

def fd_compute(contour_resampled):
    """
    Compute complex Fourier descriptors with invariance:
      - translation: subtract mean (remove DC)
      - scale: divide by |F[1]|
      - rotation: align phase by multiplying by exp(-j*angle(F[1]))
    Returns:
      Zc (complex sequence), F (normalized DFT coefficients), mu (centroid)
    """
    z = contour_resampled[:,0] + 1j*contour_resampled[:,1]
    mu = np.mean(z)
    zc = z - mu                      # translation invariance
    F = np.fft.fft(zc)

    # handle degenerate case
    if np.abs(F[1]) < 1e-8:
        scale = np.max(np.abs(F)) + 1e-8
        theta = 0.0
    else:
        scale = np.abs(F[1])
        theta = np.angle(F[1])

    Fn = F / scale                   # scale invariance
    Fn = Fn * np.exp(-1j*theta)      # rotation invariance (align first harmonic)

    return zc, Fn, mu, scale, theta

def fd_reconstruct(Fn, mu, K):
    """
    Reconstruct shape using K lowest-frequency terms (excluding DC).
    Keep symmetric low-freq: 1..K/2 and -K/2..-1.
    """
    N = Fn.shape[0]
    Fk = np.zeros_like(Fn)
    khalf = max(1, K//2)
    # copy low positive freqs
    Fk[1:1+khalf] = Fn[1:1+khalf]
    # copy low negative freqs (from the end)
    Fk[-khalf:] = Fn[-khalf:]
    z_rec = np.fft.ifft(Fk) + mu
    return np.real(z_rec), np.imag(z_rec)

def edge_density_map(edges):
    """
    8×8 map of edge pixel density (0..1).
    """
    M = np.zeros((GH, GW), np.float32)
    total = CH * CW
    for r in range(GH):
        for c in range(GW):
            y0, y1 = r*CH, (r+1)*CH
            x0, x1 = c*CW, (c+1)*CW
            block = edges[y0:y1, x0:x1]
            M[r,c] = (np.count_nonzero(block) / float(total))
    return M

def norm01(x):
    x = x.astype(np.float32)
    return (x - np.min(x)) / (np.ptp(x) + 1e-8)

# -------- main demo --------
def show_and_save_fd_demo(image_path, save_prefix="fd_demo_01", N_resample=512):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Cannot read: {image_path}")
        return
    img = ensure_size(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    contour, edges = extract_dominant_contour(img)
    if contour is None:
        print("⚠️ No valid contour found.")
        return

    cr = resample_contour(contour, n=N_resample)
    if cr is None:
        print("⚠️ Could not resample contour.")
        return

    zc, Fnorm, mu, scale, theta = fd_compute(cr)

    # ----- Plots: Original + Edges + Contour -----
    fig1, ax1 = plt.subplots(1, 3, figsize=(16, 5), dpi=110)
    ax1[0].imshow(img_rgb); ax1[0].set_title("Original + grid"); ax1[0].axis("off")
    draw_grid(ax1[0])

    ax1[1].imshow(edges, cmap='gray'); ax1[1].set_title("Canny edges"); ax1[1].axis("off")

    ax1[2].imshow(img_rgb); ax1[2].set_title("Dominant contour"); ax1[2].axis("off")
    ax1[2].plot(contour[:,0], contour[:,1], color='lime', lw=1.5)
    draw_grid(ax1[2])
    fig1.tight_layout(); plt.show()

    # ----- Descriptor magnitude spectrum (log) -----
    mags = np.abs(Fnorm)
    freqs = np.fft.fftfreq(len(Fnorm))
    fig2, ax2 = plt.subplots(1,1, figsize=(10,4), dpi=110)
    ax2.plot(np.arange(len(mags)), np.log1p(mags), lw=1.0)
    ax2.set_title("Fourier Descriptor Magnitudes (log scale)")
    ax2.set_xlabel("Frequency index"); ax2.set_ylabel("log(1 + |F_k|)")
    fig2.tight_layout(); plt.show()

    # ----- Progressive reconstructions -----
    Ks = [10, 20, 50, 100]
    fig3, axes3 = plt.subplots(1, len(Ks), figsize=(4.5*len(Ks), 5), dpi=110)
    if len(Ks) == 1: axes3 = [axes3]
    for i, K in enumerate(Ks):
        xr, yr = fd_reconstruct(Fnorm, mu, K)
        axes3[i].imshow(img_rgb); axes3[i].axis("off")
        axes3[i].plot(xr, yr, color='red', lw=1.8, label=f"K={K}")
        axes3[i].set_title(f"Reconstruction (K={K})")
        draw_grid(axes3[i])
    fig3.tight_layout(); plt.show()

    # ----- Edge-density heatmap 8×8 -----
    Emap = edge_density_map(edges)
    fig4, ax4 = plt.subplots(1,1, figsize=(6,5), dpi=110)
    im = ax4.imshow(norm01(Emap), cmap="turbo", vmin=0, vmax=1)
    ax4.set_title("Edge density (8×8)")
    ax4.set_xticks([]); ax4.set_yticks([])
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    fig4.tight_layout(); plt.show()

    # ----- Save outputs -----
    stem = os.path.splitext(os.path.basename(image_path))[0]
    base = os.path.join(OUT_DIR, f"{save_prefix}_{stem}")

    # overlays for quick export
    cv2.imwrite(base + "_original.jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(base + "_edges.png", edges)

    # save spectrum
    fig2_save, ax2s = plt.subplots(1,1, figsize=(10,4), dpi=110)
    ax2s.plot(np.arange(len(mags)), np.log1p(mags), lw=1.0)
    ax2s.set_title("Fourier Descriptor Magnitudes (log scale)")
    ax2s.set_xlabel("Frequency index"); ax2s.set_ylabel("log(1 + |F_k|)")
    fig2_save.tight_layout()
    fig2_save.savefig(base + "_fd_spectrum.png", bbox_inches="tight")
    plt.close(fig2_save)

    # save reconstructions individually
    for K in Ks:
        xr, yr = fd_reconstruct(Fnorm, mu, K)
        figK, axK = plt.subplots(1,1, figsize=(6,5), dpi=110)
        axK.imshow(img_rgb); axK.axis("off")
        axK.plot(xr, yr, color='red', lw=1.8)
        axK.set_title(f"FD reconstruction (K={K})")
        draw_grid(axK)
        figK.tight_layout()
        figK.savefig(base + f"_recon_K{K}.png", bbox_inches="tight")
        plt.close(figK)

    # save heatmap
    plt.imsave(base + "_edge_density_heatmap.png", norm01(Emap), cmap="turbo", vmin=0, vmax=1)

    # save first 128 descriptor magnitudes to CSV
    M = 128
    mags128 = mags[:M]
    df_fd = pd.DataFrame({
        "k": np.arange(M),
        "mag": mags128
    })
    df_fd.to_csv(base + "_fd_magnitudes_first128.csv", index=False)

    print(f"💾 Saved under: {OUT_DIR} with prefix {os.path.basename(base)}")

# -------- run (single image) --------
if __name__ == "__main__":
    demo_img = "26102010062.jpg"  # change to any file in IMG_DIR
    p = os.path.join(IMG_DIR, demo_img)
    if not os.path.exists(p):
        imgs = [f for f in sorted(os.listdir(IMG_DIR)) if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
        if not imgs:
            raise SystemExit(f"No images found in {IMG_DIR}")
        p = os.path.join(IMG_DIR, imgs[0])

    print("Using image:", p)
    show_and_save_fd_demo(p, save_prefix="fd_demo")