"""
preprocess_register.py
=======================
Spatial-normalisation preprocessing for the binary AD-vs-CN task.

For each CN/AD patient scan this:
  1. reorients to canonical RAS+,
  2. affine-registers the volume to the MNI152 template (dipy, MI metric with
     voxel sampling for speed),
  3. skull-strips using the MNI152 brain mask,
  4. extracts the central axial slice, resizes to 128x128, min-max normalises,
  5. caches the result as one .npy per patient.

Why: the raw ADNI N3 cohort mixes orientations, shapes, and voxel sizes, so the
CNNs were learning from spatially unaligned brains (AUC ~0.65 after only fixing
orientation). Registering every scan into a common template space is the
standard fix used by the high-accuracy ADNI literature.

RESUMABLE: results are written per-patient to cache_reg_binary/<pid>.npy. Re-runs
skip patients already cached, so an interruption never loses completed work.
When all patients are done it assembles cache_binary_registered.npz (X, y, pids)
for the trainer to load.

Run:  python preprocess_register.py
"""

import os
import json
import logging
logging.disable(logging.CRITICAL)          # silence dipy's console logger
import numpy as np
import nibabel as nib
import tensorflow as tf
from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
from nilearn.image import resample_to_img
from dipy.align import affine_registration

from train_cnn_cv import select_one_scan_per_patient, DATA_DIR, LABELS_FILE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache_reg_binary")
NPZ_OUT = os.path.join(BASE_DIR, "cache_binary_registered.npz")
IMG_SIZE = (128, 128)

# CN -> 0, AD -> 1 (MCI dropped). Matches train_cnn_binary.REMAP.
REMAP = {0: 0, 2: 1}

# Registration settings (tuned for ~14s/vol via voxel sampling; see timing runs).
REG_PIPELINE = ["center_of_mass", "translation", "rigid", "affine"]
REG_LEVEL_ITERS = [80, 40, 10]
REG_SIGMAS = [3, 1, 0]
REG_FACTORS = [4, 2, 1]
REG_NBINS = 32
REG_SAMPLING = 0.2


def _load_template():
    tmpl = load_mni152_template(resolution=2)
    static = np.asarray(tmpl.get_fdata(), dtype="float32")
    mask_img = load_mni152_brain_mask(resolution=2)
    if mask_img.shape != tmpl.shape:
        mask_img = resample_to_img(mask_img, tmpl, interpolation="nearest",
                                   force_resample=True, copy_header=True)
    mask = np.asarray(mask_img.get_fdata()) > 0
    return static, tmpl.affine, mask


def register_and_slice(path, static, static_aff, brain_mask):
    """Register one scan to MNI, skull-strip, return a (128,128,1) axial slice."""
    img = nib.as_closest_canonical(nib.load(path))
    mov = np.asarray(img.get_fdata(), dtype="float32")
    xf, _ = affine_registration(
        mov, static, moving_affine=img.affine, static_affine=static_aff,
        pipeline=REG_PIPELINE, level_iters=REG_LEVEL_ITERS,
        sigmas=REG_SIGMAS, factors=REG_FACTORS,
        nbins=REG_NBINS, sampling_proportion=REG_SAMPLING,
    )
    xf = xf * brain_mask                         # skull-strip in MNI space
    mid = xf.shape[2] // 2                        # RAS -> axis 2 is S/I
    sl = xf[:, :, mid]
    sl = tf.image.resize(sl[..., np.newaxis], IMG_SIZE).numpy()
    vmin, vmax = sl.min(), sl.max()
    sl = (sl - vmin) / (vmax - vmin + 1e-8)
    return sl.astype("float32")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    rows = select_one_scan_per_patient(DATA_DIR, LABELS_FILE)
    rows = [(p, s, REMAP[l]) for (p, s, l) in rows if l in REMAP]
    print(f"CN/AD patients to preprocess: {len(rows)}")

    static, static_aff, brain_mask = _load_template()
    print(f"MNI template {static.shape}, brain voxels={int(brain_mask.sum())}")

    done = failed = 0
    for i, (path, pid, label) in enumerate(rows, 1):
        out = os.path.join(CACHE_DIR, f"{pid}.npy")
        if os.path.exists(out):
            done += 1
            continue
        try:
            sl = register_and_slice(path, static, static_aff, brain_mask)
            np.save(out, sl)
            done += 1
        except Exception as e:
            failed += 1
            print(f"  ! {pid} failed: {e}")
        if i % 10 == 0:
            print(f"  [{i}/{len(rows)}] cached={done} failed={failed}", flush=True)

    # Assemble consolidated cache for the trainer (deterministic row order).
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
