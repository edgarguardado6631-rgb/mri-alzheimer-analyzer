"""
train_cnn_cv.py
===============
Corrected training pipeline for the ADNI 3-class (CN / MCI / AD) classifier.

This replaces the leaky single-split pipeline in `train_cnn.py`. Three concrete
methodological fixes, each of which the original code got wrong:

  1. PATIENT-LEVEL SPLITTING.
     The current cohort (Thesis_CNMCIAD_N3_841) provides ONE curated N3-corrected
     scan per patient --- 841 patients, one file each. Earlier ADNI exports mixed
     several preprocessed variants of the SAME brain (GradWarp, N3, B1 correction),
     so image-level splitting leaked the same patient into train and test and
     inflated accuracy. We keep one scan per patient and split by patient via
     StratifiedKFold, so no patient is ever in both train and test.

  2. MASK EXCLUSION.
     Some ADNI exports include binary brain MASKS (only values {0, 255}), not
     intensity MRIs. The old code loaded them as training images. We exclude
     anything whose path contains "Mask" (the N3-only cohort has none, but the
     filter keeps the pipeline safe for other exports).

  3. NO TEST-ON-VALIDATION.
     The old code used the test set as the EarlyStopping validation set and then
     reported on that same set. Here, within every fold the held-out test split
     is NEVER seen during training; early stopping uses a separate inner
     validation split carved out of the training patients only.

Evaluation: patient-level 5-fold stratified cross-validation. We report
per-fold accuracy (mean +/- std) and build a single out-of-fold (OOF) confusion
matrix + classification report over all patients (each patient is in the test
fold exactly once). A final model is then retrained on ALL patients and saved
for the web app.

Outputs written to this directory:
  - model_metrics.json     (CV summary + OOF per-class metrics; consumed by the API/frontend)
  - confusion_matrix.png   (OOF confusion matrix; thesis figure)
  - training_curves.png    (curves from a representative fold; thesis figure)
  - alzheimer_cnn_model.h5  (final model retrained on all patients)
"""

import os
import glob
import json
import numpy as np
import nibabel as nib
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "Thesis_CNMCIAD_N3_841", "ADNI")
LABELS_FILE = os.path.join(BASE_DIR, "..", "data", "cohort", "selected_cohort.csv")
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = int(os.environ.get("CV_EPOCHS", 15))
N_FOLDS = int(os.environ.get("CV_FOLDS", 5))
RANDOM_STATE = 42
LABEL_MAP = {"CN": 0, "MCI": 1, "AD": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
TARGET_NAMES = [
    "CN (Cognitively Normal)",
    "MCI (Mild Cognitive Impairment)",
    "AD (Alzheimer's Disease)",
]

# Reproducibility
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ── File selection: one clean scan per patient ──────────────────────────────
def _variant_score(path: str) -> int:
    """Rank preprocessing variants so we pick the best-quality scan per patient.
    Higher is better. Bias-field-corrected (N3) + GradWarp + B1 are preferred;
    raw 'Scaled' duplicates are mildly penalised. Masks are excluded upstream."""
    p = path.lower()
    score = 0
    if "n3" in p:
        score += 3
    if "b1_correction" in p:
        score += 2
    if "gradwarp" in p:
        score += 1
    if "scaled" in p:
        score -= 1
    return score


def select_one_scan_per_patient(data_dir, labels_file):
    """Return a list of (file_path, patient_id, label_int), one row per patient,
    masks excluded. Falls back to .nii.gz if no .nii files are found."""
    df = pd.read_csv(labels_file)

    # Resolve column names across the two ADNI export schemas we use:
    #   selected_cohort.csv : "Image ID" (int) / "Research Group" / "Subject ID"
    #   Test_2_*.csv        : "Image Data ID" (I-prefixed) / "Group" / "Subject"
    col_img = "Image ID" if "Image ID" in df.columns else "Image Data ID"
    col_grp = "Research Group" if "Research Group" in df.columns else "Group"
    col_sub = "Subject ID" if "Subject ID" in df.columns else "Subject"

    def _norm_id(v):
        # Filenames embed "...I45107"; the cohort CSV stores 45107 as an int.
        # Normalise both to the "I45107" form so they match.
        s = str(v).strip()
        return s if s.upper().startswith("I") else f"I{s}"

    id_to_label = {_norm_id(i): g for i, g in zip(df[col_img], df[col_grp])}
    id_to_subject = {_norm_id(i): s for i, s in zip(df[col_img], df[col_sub])}

    # data_dir may be a single path or a list of paths (e.g. original cohort
    # plus a newly downloaded tree). Glob .nii across all of them.
    data_dirs = data_dir if isinstance(data_dir, (list, tuple)) else [data_dir]
    files = []
    for dd in data_dirs:
        files += glob.glob(os.path.join(dd, "**/*.nii"), recursive=True)
    if not files:
        for dd in data_dirs:
            files += glob.glob(os.path.join(dd, "**/*.nii.gz"), recursive=True)

    # Exclude binary brain masks
    files = [f for f in files if "mask" not in f.lower()]

    # Group candidate scans per patient
    per_patient = {}  # subject -> list of (score, path, label_int)
    for f in files:
        fname = os.path.basename(f)
        base = fname[:-7] if fname.endswith(".nii.gz") else fname[:-4]
        image_id = _norm_id(base.split("_")[-1])
        group = id_to_label.get(image_id)
        subject = id_to_subject.get(image_id)
        if group not in LABEL_MAP or subject is None:
            continue
        per_patient.setdefault(subject, []).append(
            (_variant_score(f), f, LABEL_MAP[group])
        )

    # Pick the highest-scoring scan for each patient (deterministic tie-break)
    rows = []
    for subject, cands in per_patient.items():
        cands.sort(key=lambda t: (-t[0], t[1]))
        _, path, label = cands[0]
        rows.append((path, subject, label))

    print(f"Patients with a usable (non-mask) labelled scan: {len(rows)}")
    return rows


# ── Image loading ───────────────────────────────────────────────────────────
# Three axial levels (as fractions of S/I depth) stacked as the R,G,B channels.
# A single central slice discards most of the volume; three spaced slices give
# the 2D CNN more anatomical context (ventricles + medial-temporal region)
# without moving to a full 3D model. MUST stay in sync with api.py preprocessing.
SLICE_FRACTIONS = (0.40, 0.50, 0.60)


def _extract_slice(data, si_axis, idx):
    if si_axis == 0:
        return data[idx, :, :]
    elif si_axis == 1:
        return data[:, idx, :]
    return data[:, :, idx]


def load_slices(file_path):
    """Load three axial slices around the S/I centre (at SLICE_FRACTIONS of depth),
    resize each to IMG_SIZE, min-max normalise to [0,1], and stack them as a
    (128,128,3) image. Returns a float32 array, or None on failure."""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        axcodes = nib.aff2axcodes(img.affine)
        si_axis = next((i for i, c in enumerate(axcodes) if c in ("S", "I")), 2)
        depth = data.shape[si_axis]
        chans = []
        for frac in SLICE_FRACTIONS:
            idx = min(depth - 1, max(0, int(round(frac * (depth - 1)))))
            sl = _extract_slice(data, si_axis, idx)
            sl = tf.image.resize(sl[..., np.newaxis], IMG_SIZE).numpy()
            vmin, vmax = sl.min(), sl.max()
            sl = (sl - vmin) / (vmax - vmin + 1e-8)
            chans.append(sl[..., 0])
        return np.stack(chans, axis=-1).astype("float32")
    except Exception as e:
        print(f"  ! Failed to load {os.path.basename(file_path)}: {e}")
        return None


def build_dataset(rows):
    """Materialise X, y, and the patient id list from the selected rows."""
    X, y, pids = [], [], []
    for i, (path, subject, label) in enumerate(rows):
        sl = load_slices(path)
        if sl is None:
            continue
        X.append(sl)
        y.append(label)
        pids.append(subject)
        if (i + 1) % 25 == 0:
            print(f"  loaded {i + 1}/{len(rows)}")
    return np.array(X), np.array(y), np.array(pids)


# ── Model ───────────────────────────────────────────────────────────────────
def create_model():
    base = VGG16(weights="imagenet", include_top=False,
                 input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    unfreeze = {"block5_conv1", "block5_conv2", "block5_conv3", "block5_pool"}
    base.trainable = True
    for layer in base.layers:
        layer.trainable = layer.name in unfreeze

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = layers.RandomRotation(0.05)(inputs)
    x = layers.RandomZoom(0.1)(x)
    x = base(x, training=True)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(3, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_one(X_tr, y_tr, X_val, y_val, capture_history=False):
    """Train a fresh model on (X_tr, y_tr) with early stopping on (X_val, y_val)."""
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
    print("=== Corrected ADNI pipeline: patient-level 5-fold CV ===")
    rows = select_one_scan_per_patient(DATA_DIR, LABELS_FILE)
    if len(rows) == 0:
        print("No usable data found. Exiting.")
        raise SystemExit(1)

    print("Loading central slices...")
    X, y, pids = build_dataset(rows)
    print(f"Dataset: X={X.shape}, classes={dict(zip(*np.unique(y, return_counts=True)))}")
    print("Class counts:",
          {INV_LABEL_MAP[int(c)]: int(n) for c, n in zip(*np.unique(y, return_counts=True))})

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_acc = []
    oof_true = np.zeros(len(y), dtype=int)
    oof_pred = np.zeros(len(y), dtype=int)
    representative_history = None

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n--- Fold {fold}/{N_FOLDS} ---")
        X_tr_all, y_tr_all = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        # Inner validation split (from TRAIN ONLY) for early stopping.
        # The test fold is never used for model selection.
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_all, y_tr_all, test_size=0.2,
            random_state=RANDOM_STATE, stratify=y_tr_all,
        )

        model, hist = train_one(
            X_tr, y_tr, X_val, y_val, capture_history=(fold == 1)
        )
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

    # ── Aggregate results ──────────────────────────────────────────────────
    cv_mean, cv_std = float(np.mean(fold_acc)), float(np.std(fold_acc))
    print("\n=== Cross-validation summary ===")
    print(f"Per-fold accuracy: {[round(a, 4) for a in fold_acc]}")
    print(f"CV accuracy: {cv_mean:.4f} +/- {cv_std:.4f}")

    print("\n=== Out-of-fold classification report (all patients) ===")
    present = sorted(np.unique(oof_true))
    names = [TARGET_NAMES[i] for i in present]
    report = classification_report(
        oof_true, oof_pred, labels=present, target_names=names,
        output_dict=True, zero_division=0,
    )
    print(classification_report(oof_true, oof_pred, labels=present,
                                target_names=names, zero_division=0))

    cm = confusion_matrix(oof_true, oof_pred, labels=present)
    print("Confusion matrix (OOF):\n", cm)

    # ── Save metrics JSON (schema compatible with the existing frontend) ────
    metrics = {
        "accuracy": round(accuracy_score(oof_true, oof_pred), 4),
        "cv_accuracy_mean": round(cv_mean, 4),
        "cv_accuracy_std": round(cv_std, 4),
        "cv_folds": N_FOLDS,
        "evaluation": "patient-level 5-fold stratified CV (out-of-fold)",
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
        "model": "VGG16 fine-tuned (block5), patient-level 5-fold CV",
    }
    with open(os.path.join(BASE_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved model_metrics.json")

    # ── Confusion-matrix figure (thesis) ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(present)))
    ax.set_yticks(range(len(present)))
    short = [INV_LABEL_MAP[i] for i in present]
    ax.set_xticklabels(short)
    ax.set_yticklabels(short)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (OOF) — acc {metrics['accuracy']:.2f}")
    for i in range(len(present)):
        for j in range(len(present)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print("Saved confusion_matrix.png")

    # ── Training-curves figure (thesis) ─────────────────────────────────────
    if representative_history:
        h = representative_history
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        a1.plot(h.get("accuracy", []), label="train")
        a1.plot(h.get("val_accuracy", []), label="val")
        a1.set_title("Accuracy (fold 1)")
        a1.set_xlabel("epoch"); a1.legend()
        a2.plot(h.get("loss", []), label="train")
        a2.plot(h.get("val_loss", []), label="val")
        a2.set_title("Loss (fold 1)")
        a2.set_xlabel("epoch"); a2.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(BASE_DIR, "training_curves.png"), dpi=150)
        plt.close(fig)
        print("Saved training_curves.png")

    # ── Final model on ALL patients (for the web app) ───────────────────────
    print("\nRetraining final model on all patients for deployment...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    final_model, _ = train_one(X_tr, y_tr, X_val, y_val)
    final_model.save(os.path.join(BASE_DIR, "alzheimer_cnn_model.h5"))
    print("Saved alzheimer_cnn_model.h5")
    print("\nDone. Review model_metrics.json before updating the thesis.")
