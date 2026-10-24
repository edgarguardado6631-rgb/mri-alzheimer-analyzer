"""
train_cnn_3d.py
===============
3D CNN baseline for the ADNI 3-class (CN / MCI / AD) classifier.

Motivation
----------
The 2D pipelines (train_cnn_cv.py) plateau at ~50% CV accuracy and, critically,
fail to detect AD (recall ~0.1): a single axial slice --- or three axial levels
near the top of the brain --- misses the medial-temporal / hippocampal atrophy
that characterises Alzheimer's. This script feeds the WHOLE volume to a small 3D
CNN so the network can see that region.

Key properties (shared with the 2D pipeline):
  * PATIENT-LEVEL 5-fold stratified CV (reuses the exact selection logic from
    train_cnn_cv.select_one_scan_per_patient --- one N3 scan per patient, masks
    excluded, no leakage).
  * Inner validation split for early stopping; the test fold is never seen.
  * Out-of-fold confusion matrix + classification report over all 841 patients.

Differences from 2D:
  * No transfer learning --- VGG16 is 2D-only, so the 3D CNN trains from scratch.
  * Volumes are resampled to VOL_SHAPE (default 64^3) with linear interpolation.
  * 3D convolutions are expensive on CPU; expect minutes per epoch. Use the
    CV_FOLDS / CV_EPOCHS env vars to run a quick smoke test first.

This script writes SEPARATE artifacts so it never clobbers the deployed 2D model:
  - model_metrics_3d.json
  - confusion_matrix_3d.png
  - training_curves_3d.png
  - alzheimer_cnn_3d.h5
Promote to the canonical names (and update api.py to 3D preprocessing) only if
3D actually beats the 2D baseline.
"""

import os
import json
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the verified, leakage-free patient selection and label mapping.
from train_cnn_cv import (
    select_one_scan_per_patient,
    DATA_DIR,
    LABELS_FILE,
    LABEL_MAP,
    INV_LABEL_MAP,
    TARGET_NAMES,
)

# ── Configuration ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOL_SHAPE = (64, 64, 64)          # resampled volume (D, H, W)
BATCH_SIZE = 4
EPOCHS = int(os.environ.get("CV_EPOCHS", 15))
N_FOLDS = int(os.environ.get("CV_FOLDS", 5))
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ── Volume loading ──────────────────────────────────────────────────────────
def load_volume(file_path):
    """Load a full NIfTI volume, resample to VOL_SHAPE (linear), min-max
    normalise to [0,1]. Returns a (D,H,W,1) float32 array, or None on failure."""
    try:
        # Reorient to canonical RAS+ first: the cohort mixes ('I','P','L') and
        # ('R','A','S') scans, so without this the 3D CNN sees mirrored/rotated
        # volumes across patients and cannot learn stable features.
        img = nib.as_closest_canonical(nib.load(file_path))
        data = img.get_fdata().astype("float32")
        factors = [t / s for t, s in zip(VOL_SHAPE, data.shape)]
        vol = zoom(data, factors, order=1)
        # zoom can be off by one voxel; hard-crop/pad to the exact target.
        vol = _fit_shape(vol, VOL_SHAPE)
        vmin, vmax = vol.min(), vol.max()
        vol = (vol - vmin) / (vmax - vmin + 1e-8)
        return vol[..., np.newaxis].astype("float32")
    except Exception as e:
        print(f"  ! Failed to load {os.path.basename(file_path)}: {e}")
        return None


def _fit_shape(vol, target):
    """Center-crop or zero-pad each axis so vol matches target exactly."""
    out = vol
    for ax, (cur, tgt) in enumerate(zip(out.shape, target)):
        if cur > tgt:                      # crop
            start = (cur - tgt) // 2
            sl = [slice(None)] * out.ndim
            sl[ax] = slice(start, start + tgt)
            out = out[tuple(sl)]
        elif cur < tgt:                    # pad
            before = (tgt - cur) // 2
            after = tgt - cur - before
            pad = [(0, 0)] * out.ndim
            pad[ax] = (before, after)
            out = np.pad(out, pad, mode="constant")
    return out


def build_dataset(rows):
    """Materialise X (N,D,H,W,1), y, and patient ids from the selected rows."""
    X, y, pids = [], [], []
    for i, (path, subject, label) in enumerate(rows):
        vol = load_volume(path)
        if vol is None:
            continue
        X.append(vol)
        y.append(label)
        pids.append(subject)
        if (i + 1) % 50 == 0:
            print(f"  loaded {i + 1}/{len(rows)}")
    return np.array(X, dtype="float32"), np.array(y), np.array(pids)


# ── Model ───────────────────────────────────────────────────────────────────
def create_model():
    """Small 3D CNN trained from scratch (no ImageNet transfer for 3D)."""
    inputs = tf.keras.Input(shape=(*VOL_SHAPE, 1))
    x = layers.Conv3D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(2)(x)                       # 32^3

    x = layers.Conv3D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(2)(x)                       # 16^3

    x = layers.Conv3D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(2)(x)                       # 8^3

    x = layers.Conv3D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling3D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(3, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_one(X_tr, y_tr, X_val, y_val, capture_history=False):
    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}

    model = create_model()
    es = EarlyStopping(monitor="val_loss", patience=4,
                       restore_best_weights=True, verbose=0)
    hist = model.fit(
        X_tr, y_tr,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        class_weight=cw,
        callbacks=[es],
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    return (model, hist.history) if capture_history else (model, None)


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"=== 3D CNN pipeline: patient-level {N_FOLDS}-fold CV, vol={VOL_SHAPE} ===")
    rows = select_one_scan_per_patient(DATA_DIR, LABELS_FILE)
    if len(rows) == 0:
        print("No usable data found. Exiting.")
        raise SystemExit(1)

    print("Loading + resampling volumes (this is the slow part)...")
    X, y, pids = build_dataset(rows)
    print(f"Dataset: X={X.shape}, "
          f"classes={ {INV_LABEL_MAP[int(c)]: int(n) for c, n in zip(*np.unique(y, return_counts=True))} }")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_acc = []
    oof_true = np.zeros(len(y), dtype=int)
    oof_pred = np.zeros(len(y), dtype=int)
    representative_history = None

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n--- Fold {fold}/{N_FOLDS} ---")
        X_tr_all, y_tr_all = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_all, y_tr_all, test_size=0.2,
            random_state=RANDOM_STATE, stratify=y_tr_all,
        )

        model, hist = train_one(X_tr, y_tr, X_val, y_val, capture_history=(fold == 1))
        if fold == 1:
            representative_history = hist

        pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
        acc = accuracy_score(y_te, pred)
        fold_acc.append(acc)
        oof_true[te_idx] = y_te
        oof_pred[te_idx] = pred
        print(f"Fold {fold} accuracy: {acc:.4f}")

        del model
        tf.keras.backend.clear_session()

    cv_mean, cv_std = float(np.mean(fold_acc)), float(np.std(fold_acc))
    print("\n=== Cross-validation summary ===")
    print(f"Per-fold accuracy: {[round(a, 4) for a in fold_acc]}")
    print(f"CV accuracy: {cv_mean:.4f} +/- {cv_std:.4f}")

    print("\n=== Out-of-fold classification report (all patients) ===")
    present = sorted(np.unique(oof_true))
    names = [TARGET_NAMES[i] for i in present]
    report = classification_report(oof_true, oof_pred, labels=present,
                                   target_names=names, output_dict=True,
                                   zero_division=0)
    print(classification_report(oof_true, oof_pred, labels=present,
                                target_names=names, zero_division=0))
    cm = confusion_matrix(oof_true, oof_pred, labels=present)
    print("Confusion matrix (OOF):\n", cm)

    metrics = {
        "accuracy": round(accuracy_score(oof_true, oof_pred), 4),
        "cv_accuracy_mean": round(cv_mean, 4),
        "cv_accuracy_std": round(cv_std, 4),
        "cv_folds": N_FOLDS,
        "evaluation": "patient-level 5-fold stratified CV (out-of-fold), 3D CNN",
        "classes": [
            {
                "label": TARGET_NAMES[i],
                "precision": round(report[TARGET_NAMES[i]]["precision"], 2),
                "recall": round(report[TARGET_NAMES[i]]["recall"], 2),
                "f1": round(report[TARGET_NAMES[i]]["f1-score"], 2),
                "support": int(report[TARGET_NAMES[i]]["support"]),
            }
            for i in present
        ],
        "macro_avg": {
            "precision": round(report["macro avg"]["precision"], 2),
            "recall": round(report["macro avg"]["recall"], 2),
            "f1": round(report["macro avg"]["f1-score"], 2),
        },
        "weighted_avg": {
            "precision": round(report["weighted avg"]["precision"], 2),
            "recall": round(report["weighted avg"]["recall"], 2),
            "f1": round(report["weighted avg"]["f1-score"], 2),
        },
        "test_samples": int(len(oof_true)),
        "epochs": EPOCHS,
        "model": f"3D CNN (from scratch), vol {VOL_SHAPE}, patient-level {N_FOLDS}-fold CV",
    }
    with open(os.path.join(BASE_DIR, "model_metrics_3d.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved model_metrics_3d.json")

    # Confusion-matrix figure
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    short = [INV_LABEL_MAP[i] for i in present]
    ax.set_xticks(range(len(present))); ax.set_yticks(range(len(present)))
    ax.set_xticklabels(short); ax.set_yticklabels(short)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"3D CNN confusion (OOF) — acc {metrics['accuracy']:.2f}")
    for i in range(len(present)):
        for j in range(len(present)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "confusion_matrix_3d.png"), dpi=150)
    plt.close(fig)
    print("Saved confusion_matrix_3d.png")

    # Training-curves figure
    if representative_history:
        h = representative_history
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        a1.plot(h.get("accuracy", []), label="train")
        a1.plot(h.get("val_accuracy", []), label="val")
        a1.set_title("Accuracy (fold 1)"); a1.set_xlabel("epoch"); a1.legend()
        a2.plot(h.get("loss", []), label="train")
        a2.plot(h.get("val_loss", []), label="val")
        a2.set_title("Loss (fold 1)"); a2.set_xlabel("epoch"); a2.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(BASE_DIR, "training_curves_3d.png"), dpi=150)
        plt.close(fig)
        print("Saved training_curves_3d.png")

    # Final model on all patients (for potential deployment)
    print("\nRetraining final 3D model on all patients...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    final_model, _ = train_one(X_tr, y_tr, X_val, y_val)
    final_model.save(os.path.join(BASE_DIR, "alzheimer_cnn_3d.h5"))
    print("Saved alzheimer_cnn_3d.h5")
    print("\nDone. Compare model_metrics_3d.json against the 2D baseline before promoting.")
