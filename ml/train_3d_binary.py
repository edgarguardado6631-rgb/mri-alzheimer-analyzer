"""
train_3d_binary.py
==================
3D CNN for binary AD-vs-CN on the MNI-registered, skull-stripped volume cache
(cache_binary_registered_vol.npz, produced by preprocess_register_vol.py).

This is the culmination experiment: the 3D CNN failed earlier on UNALIGNED
volumes (AD recall ~0.16); here the inputs are registered to a common space, so
the whole-volume model should finally benefit. Same patient-level 5-fold CV,
inner-validation early stopping, OOF report + ROC-AUC as the other binary runs,
so results are directly comparable.

Outputs: model_metrics_3d_binary.json, confusion_matrix_3d_binary.png,
roc_curve_3d_binary.png, alzheimer_cnn_3d_binary.h5
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, roc_curve)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_NPZ = os.path.join(BASE_DIR, "cache_binary_registered_vol.npz")
VOL_SHAPE = (64, 64, 64)
BATCH_SIZE = 4
EPOCHS = int(os.environ.get("CV_EPOCHS", 15))
N_FOLDS = int(os.environ.get("CV_FOLDS", 5))
RANDOM_STATE = 42
TARGET_NAMES = ["CN (Cognitively Normal)", "AD (Alzheimer's Disease)"]

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def create_model():
    inputs = tf.keras.Input(shape=(*VOL_SHAPE, 1))
    x = layers.Conv3D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x); x = layers.MaxPool3D(2)(x)
    x = layers.Conv3D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPool3D(2)(x)
    x = layers.Conv3D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPool3D(2)(x)
    x = layers.Conv3D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x); x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_one(X_tr, y_tr, X_val, y_val, capture_history=False):
    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    model = create_model()
    es = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    hist = model.fit(X_tr, y_tr, epochs=EPOCHS, validation_data=(X_val, y_val),
                     class_weight=cw, callbacks=[es], batch_size=BATCH_SIZE, verbose=2)
    return (model, hist.history) if capture_history else (model, None)


if __name__ == "__main__":
    if not os.path.exists(CACHE_NPZ):
        print(f"Missing {CACHE_NPZ}. Run preprocess_register_vol.py first.")
        raise SystemExit(1)
    d = np.load(CACHE_NPZ)
    X, y, pids = d["X"], d["y"], d["pids"]
    counts = {TARGET_NAMES[int(c)]: int(n) for c, n in zip(*np.unique(y, return_counts=True))}
    print(f"=== 3D binary AD-vs-CN (registered): X={X.shape}, {counts} ===")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_acc, fold_auc = [], []
    oof_true = np.zeros(len(y), dtype=int)
    oof_pred = np.zeros(len(y), dtype=int)
    oof_prob = np.zeros(len(y), dtype=float)
    rep_hist = None

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{N_FOLDS} ---")
        X_te, y_te = X[te_idx], y[te_idx]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X[tr_idx], y[tr_idx], test_size=0.2,
            random_state=RANDOM_STATE, stratify=y[tr_idx])
        model, hist = train_one(X_tr, y_tr, X_val, y_val, capture_history=(fold == 1))
        if fold == 1:
            rep_hist = hist
        proba = model.predict(X_te, verbose=0)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fold_acc.append(accuracy_score(y_te, pred))
        fold_auc.append(roc_auc_score(y_te, proba))
        oof_true[te_idx] = y_te; oof_pred[te_idx] = pred; oof_prob[te_idx] = proba
        print(f"Fold {fold}: acc={fold_acc[-1]:.4f} AUC={fold_auc[-1]:.4f}")
        del model; tf.keras.backend.clear_session()

    oof_auc = float(roc_auc_score(oof_true, oof_prob))
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
        "evaluation": "patient-level 5-fold CV (OOF), 3D CNN, MNI-registered, binary AD vs CN",
        "classes": [{"label": TARGET_NAMES[i],
                     "precision": round(report[TARGET_NAMES[i]]["precision"], 2),
                     "recall": round(report[TARGET_NAMES[i]]["recall"], 2),
                     "f1": round(report[TARGET_NAMES[i]]["f1-score"], 2),
                     "support": int(report[TARGET_NAMES[i]]["support"])} for i in (0, 1)],
        "test_samples": int(len(oof_true)),
        "epochs": EPOCHS,
        "model": f"3D CNN, MNI-registered vol {VOL_SHAPE}, binary AD vs CN, {N_FOLDS}-fold CV",
    }
    with open(os.path.join(BASE_DIR, "model_metrics_3d_binary.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved model_metrics_3d_binary.json")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["CN", "AD"]); ax.set_yticklabels(["CN", "AD"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"3D reg. AD vs CN (OOF) — acc {metrics['accuracy']:.2f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "confusion_matrix_3d_binary.png"), dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(oof_true, oof_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {oof_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — 3D AD vs CN (registered, OOF)"); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "roc_curve_3d_binary.png"), dpi=150)
    plt.close(fig)

    print("\nRetraining final 3D binary model on all CN/AD patients...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
    final_model, _ = train_one(X_tr, y_tr, X_val, y_val)
    final_model.save(os.path.join(BASE_DIR, "alzheimer_cnn_3d_binary.h5"))
    print("Saved alzheimer_cnn_3d_binary.h5")
    print("Done.")
