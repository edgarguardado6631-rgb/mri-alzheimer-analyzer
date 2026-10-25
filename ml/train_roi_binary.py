"""
train_roi_binary.py
===================
Hippocampal-ROI 2D VGG16 for binary AD-vs-CN, on the MNI-registered volume cache.

Rationale: the whole-slice 2D model (AUC 0.69) feeds the net mostly irrelevant
cortex/ventricle. AD's signal is medial-temporal (hippocampal) atrophy. Because
every scan is registered to MNI, the hippocampus sits at known coordinates, so we
crop a fixed bilateral medial-temporal box and train only on that region.

Input: from each registered 64^3 volume, crop the hippocampal box, take 3 axial
slices spanning the band as R/G/B channels, resize to 128x128. Same 2D VGG16
(block5 fine-tuned), same patient-level 5-fold CV + ROC-AUC as the other binary
runs, so results are directly comparable.

Outputs: model_metrics_binary_roi.json, confusion_matrix_binary_roi.png,
roc_curve_binary_roi.png, alzheimer_cnn_binary_roi.h5
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, roc_curve)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess_register import _load_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_NPZ = os.path.join(BASE_DIR, "cache_binary_registered_vol.npz")
VOL_SHAPE = (64, 64, 64)
IMG_SIZE = (128, 128)
N_SLICES = 3
BATCH_SIZE = 8
EPOCHS = int(os.environ.get("CV_EPOCHS", 15))
N_FOLDS = int(os.environ.get("CV_FOLDS", 5))
RANDOM_STATE = 42
TARGET_NAMES = ["CN (Cognitively Normal)", "AD (Alzheimer's Disease)"]
TAG = os.environ.get("ROI_TAG", "")
# Bilateral medial-temporal / hippocampal box in MNI mm (generous margin).
BOX_MM = dict(x=(-40, 40), y=(-45, 2), z=(-34, 2))

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def hippo_box_idx():
    """Map the MNI-mm hippocampal box into 64^3 array indices."""
    static, aff, _ = _load_template()
    inv = np.linalg.inv(aff)
    corners = [(inv @ np.array([X, Y, Z, 1.0]))[:3]
               for X in BOX_MM["x"] for Y in BOX_MM["y"] for Z in BOX_MM["z"]]
    corners = np.array(corners)
    sc = np.array(VOL_SHAPE) / np.array(static.shape)
    lo = np.clip(np.floor(corners.min(0) * sc).astype(int), 0, np.array(VOL_SHAPE) - 1)
    hi = np.clip(np.ceil(corners.max(0) * sc).astype(int), 1, VOL_SHAPE)
    return tuple(lo), tuple(hi)


def make_roi_image(vol, box):
    """vol (64,64,64,1) -> (128,128,3): 3 axial slices through the hippo band."""
    (x0, y0, z0), (x1, y1, z1) = box
    crop = vol[x0:x1, y0:y1, z0:z1, 0]
    zs = np.linspace(0, crop.shape[2] - 1, N_SLICES).round().astype(int)
    chans = []
    for z in zs:
        sl = tf.image.resize(crop[:, :, z][..., None], IMG_SIZE).numpy()[..., 0]
        chans.append(sl)
    img = np.stack(chans, axis=-1).astype("float32")
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)


def roi_crop_to_image(crop):
    """Pre-cropped full-res ROI subvolume (dx,dy,dz) -> (128,128,3):
    N_SLICES axial slices through the band, resized and stacked."""
    zs = np.linspace(0, crop.shape[2] - 1, N_SLICES).round().astype(int)
    chans = [tf.image.resize(crop[:, :, z][..., None], IMG_SIZE).numpy()[..., 0] for z in zs]
    img = np.stack(chans, axis=-1).astype("float32")
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)


def create_model():
    base = VGG16(weights="imagenet", include_top=False,
                 input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    unfreeze = {"block5_conv1", "block5_conv2", "block5_conv3", "block5_pool"}
    base.trainable = True
    for layer in base.layers:
        layer.trainable = layer.name in unfreeze
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    # The ROI image rows are the R-L (left-right) anatomical axis, and the
    # hippocampus is bilateral, so a vertical flip maps left<->right hemisphere
    # and is a valid augmentation. Toggle with ROI_FLIP=1 (train-time only).
    if os.environ.get("ROI_FLIP") == "1":
        x = layers.RandomFlip("vertical")(inputs)
        x = layers.RandomRotation(0.05)(x)
    else:
        x = layers.RandomRotation(0.05)(inputs)
    x = layers.RandomZoom(0.1)(x)
    x = base(x, training=True)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-4), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_one(X_tr, y_tr, X_val, y_val, capture_history=False):
    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    model = create_model()
    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    hist = model.fit(X_tr, y_tr, epochs=EPOCHS, validation_data=(X_val, y_val),
                     class_weight=cw, callbacks=[es], batch_size=BATCH_SIZE, verbose=2)
    return (model, hist.history) if capture_history else (model, None)


if __name__ == "__main__":
    # Prefer the full-resolution ROI cache (pre-cropped subvolumes) if present;
    # otherwise crop the ROI from the lossy 64^3 cache.
    hires = os.path.join(BASE_DIR, "cache_binary_roi_hires.npz")
    if os.path.exists(hires):
        d = np.load(hires)
        V, y, pids = d["X"], d["y"], d["pids"]
        print(f"Using FULL-RES ROI cache: V={V.shape}")
        X = np.stack([roi_crop_to_image(v) for v in V]).astype("float32")
    elif os.path.exists(CACHE_NPZ):
        d = np.load(CACHE_NPZ)
        V, y, pids = d["X"], d["y"], d["pids"]
        box = hippo_box_idx()
        print(f"Cropping ROI from 64^3 cache; box={box}")
        X = np.stack([make_roi_image(v, box) for v in V]).astype("float32")
    else:
        print("No ROI cache found. Run preprocess_register_vol.py or _roi.py first.")
        raise SystemExit(1)
    counts = {TARGET_NAMES[int(c)]: int(n) for c, n in zip(*np.unique(y, return_counts=True))}
    print(f"=== ROI 2D binary AD-vs-CN: X={X.shape}, {counts} ===")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_acc, fold_auc = [], []
    oof_true = np.zeros(len(y), dtype=int)
    oof_pred = np.zeros(len(y), dtype=int)
    oof_prob = np.zeros(len(y), dtype=float)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{N_FOLDS} ---")
        X_te, y_te = X[te_idx], y[te_idx]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X[tr_idx], y[tr_idx], test_size=0.2,
            random_state=RANDOM_STATE, stratify=y[tr_idx])
        model, _ = train_one(X_tr, y_tr, X_val, y_val)
        proba = model.predict(X_te, verbose=0)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fold_acc.append(accuracy_score(y_te, pred))
        fold_auc.append(roc_auc_score(y_te, proba))
        oof_true[te_idx] = y_te; oof_pred[te_idx] = pred; oof_prob[te_idx] = proba
        print(f"Fold {fold}: acc={fold_acc[-1]:.4f} AUC={fold_auc[-1]:.4f}")
        del model; tf.keras.backend.clear_session()

    oof_auc = float(roc_auc_score(oof_true, oof_prob))
    # Keep every subject's out-of-fold prediction, so later analyses (subgroups,
    # calibration, bootstrap intervals) need no retraining.
    import pandas as pd
    fold_of = np.zeros(len(y), dtype=int)
    for k, (_, te) in enumerate(skf.split(X, y), 1):
        fold_of[te] = k
    pd.DataFrame({"subject": pids, "y_true": oof_true, "prob_ad": oof_prob, "fold": fold_of}).to_csv(
        os.path.join(BASE_DIR, "results", "metrics", f"oof_binary_roi{TAG}.csv"), index=False)
    print("\n=== CV summary ===")
    print(f"Per-fold accuracy: {[round(a,4) for a in fold_acc]}")
    print(f"Per-fold AUC:      {[round(a,4) for a in fold_auc]}")
    print(f"CV accuracy: {np.mean(fold_acc):.4f} +/- {np.std(fold_acc):.4f}")
    print(f"CV AUC:      {np.mean(fold_auc):.4f} +/- {np.std(fold_auc):.4f}  (OOF AUC={oof_auc:.4f})")
    print("\n=== OOF classification report ===")
    report = classification_report(oof_true, oof_pred, labels=[0, 1],
                                   target_names=TARGET_NAMES, output_dict=True, zero_division=0)
    print(classification_report(oof_true, oof_pred, labels=[0, 1],
                                target_names=TARGET_NAMES, zero_division=0))
    cm = confusion_matrix(oof_true, oof_pred, labels=[0, 1])
    print("Confusion matrix (OOF):\n", cm)

    metrics = {
        "accuracy": round(accuracy_score(oof_true, oof_pred), 4),
        "cv_accuracy_mean": round(float(np.mean(fold_acc)), 4),
        "cv_accuracy_std": round(float(np.std(fold_acc)), 4),
        "roc_auc": round(oof_auc, 4),
        "cv_auc_mean": round(float(np.mean(fold_auc)), 4),
        "cv_auc_std": round(float(np.std(fold_auc)), 4),
        "cv_folds": N_FOLDS,
        "evaluation": "patient-level 5-fold CV (OOF), 2D VGG16, MNI-registered hippocampal ROI, binary AD vs CN",
        "classes": [{"label": TARGET_NAMES[i],
                     "precision": round(report[TARGET_NAMES[i]]["precision"], 2),
                     "recall": round(report[TARGET_NAMES[i]]["recall"], 2),
                     "f1": round(report[TARGET_NAMES[i]]["f1-score"], 2),
                     "support": int(report[TARGET_NAMES[i]]["support"])} for i in (0, 1)],
        "test_samples": int(len(oof_true)),
        "epochs": EPOCHS,
        "roi_box_mm": BOX_MM,
        "model": "2D VGG16 fine-tuned, MNI-registered hippocampal ROI (3-slice), binary AD vs CN",
    }
    with open(os.path.join(BASE_DIR, f"model_metrics_binary_roi{TAG}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved model_metrics_binary_roi.json")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["CN", "AD"]); ax.set_yticklabels(["CN", "AD"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"ROI AD vs CN (OOF) — acc {metrics['accuracy']:.2f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, f"confusion_matrix_binary_roi{TAG}.png"), dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(oof_true, oof_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {oof_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — hippocampal ROI AD vs CN (OOF)"); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, f"roc_curve_binary_roi{TAG}.png"), dpi=150)
    plt.close(fig)

    if os.environ.get("SKIP_FINAL") == "1":
        print("SKIP_FINAL=1: cross-validation only, no final model.")
        raise SystemExit(0)
    print("\nRetraining final ROI model on all CN/AD patients...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
    final_model, _ = train_one(X_tr, y_tr, X_val, y_val)
    final_model.save(os.path.join(BASE_DIR, f"alzheimer_cnn_binary_roi{TAG}.h5"))
    print("Saved alzheimer_cnn_binary_roi.h5")
    print("Done.")
