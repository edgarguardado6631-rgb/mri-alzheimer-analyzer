"""
train_roi_fullregion.py
=======================
Same hippocampal ROI, same VGG16 and same patient-level folds as
train_roi_binary.py, but the network sees all 18 axial slices of the region
instead of 3.

The 18 slices are grouped into 6 RGB images of 3 consecutive slices. One shared
VGG16 (ImageNet weights, block5 fine-tuned) encodes each image, and the 6
feature maps are concatenated before the same Dense 64 + Dropout head. The
augmentation is applied to the whole 18-slice stack, so every slice of a subject
gets the same rotation and zoom.

Cross-validation only (no final model). Outputs:
results/metrics/model_metrics_binary_roi_full{TAG}.json and
results/metrics/oof_binary_roi_full{TAG}.csv
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MET_DIR = os.path.join(BASE_DIR, "results", "metrics")
IMG_SIZE = (128, 128)
GROUP = 3                      # consecutive slices per RGB image
BATCH_SIZE = 8
EPOCHS = int(os.environ.get("CV_EPOCHS", 15))
N_FOLDS = int(os.environ.get("CV_FOLDS", 5))
RANDOM_STATE = 42
TAG = os.environ.get("ROI_TAG", "")

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def roi_crop_to_stack(crop):
    """ROI subvolume (dx,dy,dz) -> (128,128,dz) with every axial slice, min-max
    normalised over the whole subject (keeps intensity comparable across slices)."""
    sl = tf.image.resize(crop[..., None].transpose(2, 0, 1, 3), IMG_SIZE).numpy()[..., 0]
    img = sl.transpose(1, 2, 0).astype("float32")
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)


def create_model(n_slices):
    n_groups = n_slices // GROUP
    base = VGG16(weights="imagenet", include_top=False,
                 input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    unfreeze = {"block5_conv1", "block5_conv2", "block5_conv3", "block5_pool"}
    base.trainable = True
    for layer in base.layers:
        layer.trainable = layer.name in unfreeze
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], n_slices))
    x = layers.RandomRotation(0.05)(inputs)
    x = layers.RandomZoom(0.1)(x)
    # (128,128,18) -> (6,128,128,3): slices [0,1,2], [3,4,5], ... as RGB images
    x = layers.Reshape((IMG_SIZE[0], IMG_SIZE[1], n_groups, GROUP))(x)
    x = layers.Permute((3, 1, 2, 4))(x)
    x = layers.TimeDistributed(base)(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-4), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_one(X_tr, y_tr, X_val, y_val):
    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    model = create_model(X_tr.shape[-1])
    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(X_tr, y_tr, epochs=EPOCHS, validation_data=(X_val, y_val),
              class_weight=cw, callbacks=[es], batch_size=BATCH_SIZE, verbose=2)
    return model


if __name__ == "__main__":
    d = np.load(os.path.join(BASE_DIR, "cache_binary_roi_hires.npz"))
    V, y, pids = d["X"], d["y"], d["pids"]
    n_keep = V.shape[3] - V.shape[3] % GROUP
    X = np.stack([roi_crop_to_stack(v[:, :, :n_keep]) for v in V]).astype("float32")
    print(f"=== Full-region ROI, AD vs CN: X={X.shape}, AD={int(y.sum())}, CN={int((y == 0).sum())} ===")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_acc, fold_auc = [], []
    oof_prob = np.zeros(len(y))
    fold_of = np.zeros(len(y), dtype=int)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{N_FOLDS} ---")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X[tr_idx], y[tr_idx], test_size=0.2,
            random_state=RANDOM_STATE, stratify=y[tr_idx])
        model = train_one(X_tr, y_tr, X_val, y_val)
        proba = model.predict(X[te_idx], batch_size=BATCH_SIZE, verbose=0)[:, 1]
        fold_acc.append(accuracy_score(y[te_idx], (proba >= 0.5).astype(int)))
        fold_auc.append(roc_auc_score(y[te_idx], proba))
        oof_prob[te_idx] = proba
        fold_of[te_idx] = fold
        print(f"Fold {fold}: acc={fold_acc[-1]:.4f} AUC={fold_auc[-1]:.4f}", flush=True)
        del model; tf.keras.backend.clear_session()

    pred = (oof_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics = {
        "model": "VGG16 shared over 6 RGB groups of 3 axial slices (all 18 ROI slices), binary AD vs CN",
        "cv_auc_mean": round(float(np.mean(fold_auc)), 4),
        "cv_auc_std": round(float(np.std(fold_auc)), 4),
        "cv_accuracy_mean": round(float(np.mean(fold_acc)), 4),
        "cv_accuracy_std": round(float(np.std(fold_acc)), 4),
        "fold_auc": [round(a, 4) for a in fold_auc],
        "oof_auc": round(float(roc_auc_score(y, oof_prob)), 4),
        "sensitivity": round(tp / (tp + fn), 4),
        "specificity": round(tn / (tn + fp), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "epochs": EPOCHS, "cv_folds": N_FOLDS,
    }
    pd.DataFrame({"subject": pids, "y_true": y, "prob_ad": oof_prob, "fold": fold_of}).to_csv(
        os.path.join(MET_DIR, f"oof_binary_roi_full{TAG}.csv"), index=False)
    with open(os.path.join(MET_DIR, f"model_metrics_binary_roi_full{TAG}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n=== CV summary ===")
    print(f"Per-fold AUC: {metrics['fold_auc']}")
    print(f"CV AUC:      {metrics['cv_auc_mean']:.4f} +/- {metrics['cv_auc_std']:.4f}  (OOF AUC={metrics['oof_auc']:.4f})")
    print(f"CV accuracy: {metrics['cv_accuracy_mean']:.4f} +/- {metrics['cv_accuracy_std']:.4f}")
