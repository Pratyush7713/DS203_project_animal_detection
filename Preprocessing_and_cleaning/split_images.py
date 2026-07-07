# Phase 1: Split images (by image, not cells) into train / val / test with stratification

import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# -------- paths --------
PREPROCESSED_DIR = "/home/pratyush/Desktop/DS_Project/data/preprocessed"
LABELS_CSV       = "/home/pratyush/Desktop/DS_Project/data/labels.csv"
SPLIT_DIR        = "/home/pratyush/Desktop/DS_Project/data/splits"  # outputs here

os.makedirs(SPLIT_DIR, exist_ok=True)

# -------- load labels --------
df = pd.read_csv(LABELS_CSV)
cols = ["image"] + [f"c{i:02d}" for i in range(1, 65)]
missing = [c for c in cols if c not in df.columns]
if missing:
    raise ValueError(f"labels.csv is missing columns: {missing}")

# keep only rows whose image file exists
df["path"] = df["image"].apply(lambda f: os.path.join(PREPROCESSED_DIR, f))
df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)

# sanity check count (you mentioned 424 images)
print(f"Found {len(df)} labeled images with existing files.")

# -------- stratification target (per-image positive count) --------
# number of positive cells in each image
pos_cols = [f"c{i:02d}" for i in range(1, 65)]
df["pos_count"] = df[pos_cols].sum(axis=1)

# bin the pos_count to build a stratification label that's not too sparse
# try quantile bins (5 bins); if data has many ties, fallback to fixed bins
def make_strata(s):
    try:
        return pd.qcut(s, q=5, duplicates="drop")
    except Exception:
        # fallback: fixed bins (tweak if needed)
        bins = [-0.1, 0, 2, 5, 10, 64]
        return pd.cut(s, bins=bins, include_lowest=True)

df["strata"] = make_strata(df["pos_count"])

# -------- split 70 / 15 / 15 by image --------
rng = 42

# train vs temp (val+test)
df_train, df_temp = train_test_split(
    df, test_size=0.30, random_state=rng, stratify=df["strata"]
)

# val vs test (split temp 50/50)
df_val, df_test = train_test_split(
    df_temp, test_size=0.50, random_state=rng, stratify=df_temp["strata"]
)

print(f"Split sizes (images): train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

# -------- save label CSVs for each split --------
def save_split(split_df, name):
    out_csv = os.path.join(SPLIT_DIR, f"{name}_labels.csv")
    split_df[cols].to_csv(out_csv, index=False)
    with open(os.path.join(SPLIT_DIR, f"{name}_images.txt"), "w") as f:
        for fn in split_df["image"]:
            f.write(f"{fn}\n")
    print(f"Saved {name}: {out_csv}")

save_split(df_train, "train")
save_split(df_val, "val")
save_split(df_test, "test")

# -------- (optional) copy images into split folders for convenience --------
COPY_IMAGES = False  # set True if you want physical split folders

if COPY_IMAGES:
    for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        out_dir = os.path.join(PREPROCESSED_DIR, name)
        os.makedirs(out_dir, exist_ok=True)
        for fp in split_df["path"]:
            shutil.copy2(fp, os.path.join(out_dir, os.path.basename(fp)))
        print(f"Copied {len(split_df)} images to {out_dir}")
