"""
train_cnn_binary.py
===================
Binary AD-vs-CN classifier (MCI excluded).

Rationale
---------
2D single-slice, 2D multi-slice, and volumetric 3D pipelines all plateau at
~50% CV accuracy on the 3-class (CN/MCI/AD) task, and all fail the same way:
they collapse toward the ambiguous MCI middle class and miss AD (recall ~0.15).
The bottleneck is the 3-way framing, not the input or the architecture. This
script drops MCI and learns the clinically decisive, separable distinction ---
cognitively normal vs. Alzheimer's --- which is also the framing most of the
cited literature uses (so the comparison becomes fair).

Architecture: the proven single-slice 2D VGG16 (block5 fine-tuned). Transfer
learning is the key asset here --- after dropping MCI only ~544 patients remain,
too few to train a competitive network from scratch.

Method (unchanged from the 3-class pipeline, so results stay comparable):
  * ONE N3 scan per patient, masks excluded, patient-level 5-fold stratified CV.
  * Inner validation split for early stopping; test fold never seen in training.
  * Out-of-fold confusion matrix + classification report over all CN/AD patients.
  * Adds ROC-AUC (the standard binary metric) on out-of-fold AD probabilities.

Separate artifacts (never clobbers the deployed 3-class model):
  - model_metrics_binary.json
  - confusion_matrix_binary.png
  - training_curves_binary.png
  - roc_curve_binary.png
  - alzheimer_cnn_binary.h5
"""

import os
import json
import numpy as np
import nibabel as nib
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

# Reuse verified, leakage-free patient selection (3-class label map + data paths).
from train_cnn_cv import select_one_scan_per_patient, DATA_DIR, LABELS_FILE

# ── Configuration ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = int(os.environ.get("CV_EPOCHS", 15))
N_FOLDS = int(os.environ.get("CV_FOLDS", 5))
RANDOM_STATE = 42

# select_one_scan_per_patient uses LABEL_MAP {CN:0, MCI:1, AD:2}. Keep CN & AD,
# drop MCI, and remap to a binary target: CN -> 0, AD -> 1.
SRC_CN, SRC_AD = 0, 2
REMAP = {SRC_CN: 0, SRC_AD: 1}
TARGET_NAMES = ["CN (Cognitively Normal)", "AD (Alzheimer's Disease)"]

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ── Image loading (single central axial slice) ──────────────────────────────
def load_slice(file_path):
    """Central axial slice (S/I axis), resized, min-max normalised to [0,1].
    Returns a (128,128,1) float32 array, or None on failure."""
    try:
        img = nib.load(file_path)
        # Reorient to canonical RAS+ so every scan shares one orientation.
        # The cohort mixes ('I','P','L') and ('R','A','S') scans; without this
        # the "central axial slice" shows mirrored/rotated anatomy across
        # patients and no model can learn.
        img = nib.as_closest_canonical(img)
        data = img.get_fdata()
        axcodes = nib.aff2axcodes(img.affine)
        si_axis = next((i for i, c in enumerate(axcodes) if c in ("S", "I")), 2)
        mid = data.shape[si_axis] // 2
        if si_axis == 0:
            sl = data[mid, :, :]
        elif si_axis == 1:
            sl = data[:, mid, :]
        else:
            sl = data[:, :, mid]
        sl = tf.image.resize(sl[..., np.newaxis], IMG_SIZE).numpy()
        vmin, vmax = sl.min(), sl.max()
        sl = (sl - vmin) / (vmax - vmin + 1e-8)
        return sl.astype("float32")
    except Exception as e:
        print(f"  ! Failed to load {os.path.basename(file_path)}: {e}")
        return None


def build_dataset(rows):
    X, y, pids = [], [], []
    for i, (path, subject, label) in enumerate(rows):
        sl = load_slice(path)
        if sl is None:
            continue
        X.append(sl)
        y.append(label)
        pids.append(subject)
        if (i + 1) % 50 == 0:
            print(f"  loaded {i + 1}/{len(rows)}")
    return np.array(X), np.array(y), np.array(pids)


# ── Model (single-slice VGG16, block5 fine-tuned, 2-class head) ──────────────
def create_model():
    base = VGG16(weights="imagenet", include_top=False,
                 input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    unfreeze = {"block5_conv1", "block5_conv2", "block5_conv3", "block5_pool"}
    base.trainable = True
    for layer in base.layers:
        layer.trainable = layer.name in unfreeze

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
    x = layers.Concatenate(axis=-1)([inputs, inputs, inputs])
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    x = base(x, training=True)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_one(X_tr, y_tr, X_val, y_val, capture_history=False):
    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}

    model = create_model()
    es = EarlyStopping(monitor="val_loss", patience=3,
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
    print(f"=== Binary AD-vs-CN pipeline: patient-level {N_FOLDS}-fold CV ===")

    # Prefer the MNI-registered, skull-stripped cache if preprocessing has run.
    # Its slices are already reoriented/registered/normalised, so we load X
    # directly and skip build_dataset. Outputs get a "_registered" suffix so the
    # pre-registration baseline artifacts are preserved for comparison.
    cache_npz = os.path.join(BASE_DIR, "cache_binary_registered.npz")
    if os.path.exists(cache_npz):
        print(f"Loading MNI-registered cache: {cache_npz}")
        d = np.load(cache_npz)
        X, y, pids = d["X"], d["y"], d["pids"]
        SUFFIX = "_registered"
    else:
        rows = select_one_scan_per_patient(DATA_DIR, LABELS_FILE)
        rows = [(p, s, REMAP[l]) for (p, s, l) in rows if l in REMAP]
        if len(rows) == 0:
            print("No CN/AD data found. Exiting.")
            raise SystemExit(1)
        print("Loading central slices (CN + AD only)...")
        X, y, pids = build_dataset(rows)
        SUFFIX = ""

    counts = {TARGET_NAMES[int(c)]: int(n) for c, n in zip(*np.unique(y, return_counts=True))}
    print(f"Dataset: X={X.shape}, classes={counts}, suffix='{SUFFIX}'")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_acc, fold_auc = [], []
    oof_true = np.zeros(len(y), dtype=int)
    oof_pred = np.zeros(len(y), dtype=int)
    oof_prob = np.zeros(len(y), dtype=float)   # P(AD) for AUC
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

        proba = model.predict(X_te, verbose=0)[:, 1]
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y_te, pred)
        auc = roc_auc_score(y_te, proba)
        fold_acc.append(acc)
        fold_auc.append(auc)
        oof_true[te_idx] = y_te
        oof_pred[te_idx] = pred
        oof_prob[te_idx] = proba
        print(f"Fold {fold}: accuracy={acc:.4f}  AUC={auc:.4f}")

        del model
        tf.keras.backend.clear_session()

    cv_acc_mean, cv_acc_std = float(np.mean(fold_acc)), float(np.std(fold_acc))
    cv_auc_mean, cv_auc_std = float(np.mean(fold_auc)), float(np.std(fold_auc))
    oof_auc = float(roc_auc_score(oof_true, oof_prob))
    print("\n=== Cross-validation summary ===")
    print(f"Per-fold accuracy: {[round(a, 4) for a in fold_acc]}")
    print(f"Per-fold AUC:      {[round(a, 4) for a in fold_auc]}")
    print(f"CV accuracy: {cv_acc_mean:.4f} +/- {cv_acc_std:.4f}")
    print(f"CV AUC:      {cv_auc_mean:.4f} +/- {cv_auc_std:.4f}   (OOF AUC={oof_auc:.4f})")

    print("\n=== Out-of-fold classification report ===")
    report = classification_report(oof_true, oof_pred, labels=[0, 1],
                                   target_names=TARGET_NAMES, output_dict=True,
                                   zero_division=0)
    print(classification_report(oof_true, oof_pred, labels=[0, 1],
                                target_names=TARGET_NAMES, zero_division=0))
    cm = confusion_matrix(oof_true, oof_pred, labels=[0, 1])
    print("Confusion matrix (OOF):\n", cm)

    metrics = {
        "accuracy": round(accuracy_score(oof_true, oof_pred), 4),
        "cv_accuracy_mean": round(cv_acc_mean, 4),
        "cv_accuracy_std": round(cv_acc_std, 4),
        "roc_auc": round(oof_auc, 4),
        "cv_auc_mean": round(cv_auc_mean, 4),
        "cv_auc_std": round(cv_auc_std, 4),
        "cv_folds": N_FOLDS,
        "evaluation": "patient-level 5-fold stratified CV (out-of-fold), binary AD vs CN",
        "classes": [
            {
                "label": TARGET_NAMES[i],
                "precision": round(report[TARGET_NAMES[i]]["precision"], 2),
                "recall": round(report[TARGET_NAMES[i]]["recall"], 2),
                "f1": round(report[TARGET_NAMES[i]]["f1-score"], 2),
                "support": int(report[TARGET_NAMES[i]]["support"]),
            }
            for i in (0, 1)
        ],
        "macro_avg": {
            "precision": round(report["macro avg"]["precision"], 2),
            "recall": round(report["macro avg"]["recall"], 2),
            "f1": round(report["macro avg"]["f1-score"], 2),
        },
        "test_samples": int(len(oof_true)),
        "epochs": EPOCHS,
        "model": "VGG16 fine-tuned (block5), single-slice, binary AD vs CN, patient-level 5-fold CV",
    }
    with open(os.path.join(BASE_DIR, f"model_metrics_binary{SUFFIX}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved model_metrics_binary.json")

    # Confusion-matrix figure
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["CN", "AD"]); ax.set_yticklabels(["CN", "AD"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"AD vs CN confusion (OOF) — acc {metrics['accuracy']:.2f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, f"confusion_matrix_binary{SUFFIX}.png"), dpi=150)
    plt.close(fig)
    print("Saved confusion_matrix_binary.png")

    # ROC curve (OOF)
    fpr, tpr, _ = roc_curve(oof_true, oof_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {oof_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — AD vs CN (out-of-fold)"); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, f"roc_curve_binary{SUFFIX}.png"), dpi=150)
    plt.close(fig)
    print("Saved roc_curve_binary.png")

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
        fig.savefig(os.path.join(BASE_DIR, f"training_curves_binary{SUFFIX}.png"), dpi=150)
        plt.close(fig)
        print("Saved training_curves_binary.png")

    # Final model on all CN/AD patients
    print("\nRetraining final binary model on all CN/AD patients...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    final_model, _ = train_one(X_tr, y_tr, X_val, y_val)
    final_model.save(os.path.join(BASE_DIR, f"alzheimer_cnn_binary{SUFFIX}.h5"))
    print("Saved alzheimer_cnn_binary.h5")
    print("\nDone. Compare model_metrics_binary.json against the 3-class baselines.")
