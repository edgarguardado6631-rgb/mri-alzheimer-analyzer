"""
preprocess_register_roi.py
==========================
Cache the hippocampal ROI at FULL template resolution (2mm MNI, 99x117x95 grid),
not the lossy 64^3 downsample. The ROI experiment that lifted AUC to 0.79 cropped
a 27x14x13 box from the 64^3 cache and upsampled it 5x (blurry). Here we register
to the 2mm template and crop the same MNI-mm box at full resolution (~40x24x18),
giving ~3x the in-plane detail.

Resumable: one .npy per patient in cache_reg_binary_roi/; assembles
cache_binary_roi_hires.npz (X=(N, dx, dy, dz), y, pids).
"""

import os
import logging
logging.disable(logging.CRITICAL)
import numpy as np
import nibabel as nib

from preprocess_register import (
    _load_template, REMAP, REG_PIPELINE, REG_LEVEL_ITERS,
    REG_SIGMAS, REG_FACTORS, REG_NBINS, REG_SAMPLING,
)
from dipy.align import affine_registration
from train_cnn_cv import select_one_scan_per_patient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Merged cohort: original 841-subject tree + the newly downloaded 1,744-subject
# tree, with the merged Image-ID->group manifest. The per-subject cache means the
# 544 already-registered AD/CN subjects are skipped; only the new ones register.
DATA_DIRS = [
    os.path.join(BASE_DIR, "..", "data", "Thesis_CNMCIAD_N3_841", "ADNI"),
    os.path.join(BASE_DIR, "..", "data", "new", "images", "ADNI"),
]
LABELS_FILE = os.path.join(BASE_DIR, "..", "data", "cohort", "merged_cohort.csv")
CACHE_DIR = os.path.join(BASE_DIR, "cache_reg_binary_roi")
NPZ_OUT = os.path.join(BASE_DIR, "cache_binary_roi_hires.npz")
BOX_MM = dict(x=(-40, 40), y=(-45, 2), z=(-34, 2))  # matches train_roi_binary


def full_res_box(static_shape, aff):
    inv = np.linalg.inv(aff)
    corners = [(inv @ np.array([X, Y, Z, 1.0]))[:3]
               for X in BOX_MM["x"] for Y in BOX_MM["y"] for Z in BOX_MM["z"]]
    corners = np.array(corners)
    lo = np.clip(np.floor(corners.min(0)).astype(int), 0, np.array(static_shape) - 1)
    hi = np.clip(np.ceil(corners.max(0)).astype(int), 1, static_shape)
    return tuple(lo), tuple(hi)


def register_and_crop(path, static, static_aff, brain_mask, box):
    img = nib.as_closest_canonical(nib.load(path))
    mov = np.asarray(img.get_fdata(), dtype="float32")
    xf, _ = affine_registration(
        mov, static, moving_affine=img.affine, static_affine=static_aff,
        pipeline=REG_PIPELINE, level_iters=REG_LEVEL_ITERS,
        sigmas=REG_SIGMAS, factors=REG_FACTORS,
        nbins=REG_NBINS, sampling_proportion=REG_SAMPLING,
    )
    xf = xf * brain_mask
    (x0, y0, z0), (x1, y1, z1) = box
    crop = xf[x0:x1, y0:y1, z0:z1].astype("float32")
    mn, mx = crop.min(), crop.max()
    return ((crop - mn) / (mx - mn + 1e-8)).astype("float32")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    rows = select_one_scan_per_patient(DATA_DIRS, LABELS_FILE)
    rows = [(p, s, REMAP[l]) for (p, s, l) in rows if l in REMAP]
    static, static_aff, brain_mask = _load_template()
    box = full_res_box(static.shape, static_aff)
    print(f"Full-res ROI box {box} size={np.array(box[1])-np.array(box[0])}  patients={len(rows)}")

    done = failed = 0
    for i, (path, pid, label) in enumerate(rows, 1):
        out = os.path.join(CACHE_DIR, f"{pid}.npy")
        if os.path.exists(out):
            done += 1
            continue
        try:
            np.save(out, register_and_crop(path, static, static_aff, brain_mask, box))
            done += 1
        except Exception as e:
            failed += 1
            print(f"  ! {pid} failed: {e}")
        if i % 10 == 0:
            print(f"  [{i}/{len(rows)}] cached={done} failed={failed}", flush=True)

    X, y, pids = [], [], []
    for path, pid, label in rows:
        out = os.path.join(CACHE_DIR, f"{pid}.npy")
        if os.path.exists(out):
            X.append(np.load(out)); y.append(label); pids.append(pid)
    X = np.array(X, dtype="float32"); y = np.array(y); pids = np.array(pids)
    np.savez_compressed(NPZ_OUT, X=X, y=y, pids=pids)
    counts = {int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))}
    print(f"\nAssembled {NPZ_OUT}: X={X.shape}, labels(0=CN,1=AD)={counts}, failed={failed}")


if __name__ == "__main__":
    main()
