"""
preprocess_register_vol.py
==========================
Like preprocess_register.py, but caches the full registered + skull-stripped
VOLUME (downsampled to 64^3) per patient, for the 3D CNN on aligned data.

Resumable: one .npy per patient in cache_reg_binary_vol/; re-runs skip finished
ones. Assembles cache_binary_registered_vol.npz (X=(N,64,64,64,1), y, pids).
"""

import os
import logging
logging.disable(logging.CRITICAL)
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

from preprocess_register import (
    _load_template, REMAP, REG_PIPELINE, REG_LEVEL_ITERS,
    REG_SIGMAS, REG_FACTORS, REG_NBINS, REG_SAMPLING,
)
from dipy.align import affine_registration
from train_cnn_cv import select_one_scan_per_patient, DATA_DIR, LABELS_FILE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache_reg_binary_vol")
NPZ_OUT = os.path.join(BASE_DIR, "cache_binary_registered_vol.npz")
VOL_SHAPE = (64, 64, 64)


def register_volume(path, static, static_aff, brain_mask):
    img = nib.as_closest_canonical(nib.load(path))
    mov = np.asarray(img.get_fdata(), dtype="float32")
    xf, _ = affine_registration(
        mov, static, moving_affine=img.affine, static_affine=static_aff,
        pipeline=REG_PIPELINE, level_iters=REG_LEVEL_ITERS,
        sigmas=REG_SIGMAS, factors=REG_FACTORS,
        nbins=REG_NBINS, sampling_proportion=REG_SAMPLING,
    )
    xf = xf * brain_mask
    factors = [t / s for t, s in zip(VOL_SHAPE, xf.shape)]
    vol = zoom(xf, factors, order=1)
    # exact shape guard
    sl = tuple(slice(0, t) for t in VOL_SHAPE)
    vol = vol[sl]
    pad = [(0, max(0, t - c)) for t, c in zip(VOL_SHAPE, vol.shape)]
    vol = np.pad(vol, pad, mode="constant")
    vmin, vmax = vol.min(), vol.max()
    vol = (vol - vmin) / (vmax - vmin + 1e-8)
    return vol[..., np.newaxis].astype("float32")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    rows = select_one_scan_per_patient(DATA_DIR, LABELS_FILE)
    rows = [(p, s, REMAP[l]) for (p, s, l) in rows if l in REMAP]
    print(f"CN/AD patients: {len(rows)}")
    static, static_aff, brain_mask = _load_template()

    done = failed = 0
    for i, (path, pid, label) in enumerate(rows, 1):
        out = os.path.join(CACHE_DIR, f"{pid}.npy")
        if os.path.exists(out):
            done += 1
            continue
        try:
            np.save(out, register_volume(path, static, static_aff, brain_mask))
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
