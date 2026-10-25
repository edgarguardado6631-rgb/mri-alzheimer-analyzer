"""
build_ensemble_metrics.py
=========================
Writes models/final_metrics.json for the deployed model: the average of three
full-region VGG16 networks (train_roi_fullregion.py).

The cross-validated estimate averages the out-of-fold probabilities of the three
CV runs (same patient-level folds), which is exactly what the deployed ensemble
does with its three members.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

BASE = os.path.dirname(os.path.abspath(__file__))
MET = os.path.join(BASE, "results", "metrics")
RUNS = ["oof_binary_roi_full.csv", "oof_binary_roi_full_rep2.csv", "oof_binary_roi_full_rep3.csv"]


def main():
    frames = [pd.read_csv(os.path.join(MET, r)).set_index("subject") for r in RUNS]
    ref = frames[0]
    for f in frames[1:]:
        assert (f.loc[ref.index, "y_true"] == ref["y_true"]).all()
        assert (f.loc[ref.index, "fold"] == ref["fold"]).all()
    y = ref["y_true"].to_numpy()
    fold = ref["fold"].to_numpy()
    prob = np.mean([f.loc[ref.index, "prob_ad"].to_numpy() for f in frames], axis=0)
    pred = (prob >= 0.5).astype(int)

    fold_auc = [roc_auc_score(y[fold == k], prob[fold == k]) for k in sorted(set(fold))]
    fold_acc = [accuracy_score(y[fold == k], pred[fold == k]) for k in sorted(set(fold))]
    member_auc = [json.load(open(os.path.join(MET, n)))["cv_auc_mean"] for n in (
        "model_metrics_binary_roi_full.json", "model_metrics_binary_roi_full_rep2.json",
        "model_metrics_binary_roi_full_rep3.json")]
    p, r, f1, s = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    names = ["CN (Cognitively Normal)", "AD (Alzheimer's Disease)"]

    metrics = {
        "accuracy": round(accuracy_score(y, pred), 4),
        "cv_accuracy_mean": round(float(np.mean(fold_acc)), 4),
        "cv_accuracy_std": round(float(np.std(fold_acc)), 4),
        "roc_auc": round(roc_auc_score(y, prob), 4),
        "cv_auc_mean": round(float(np.mean(fold_auc)), 4),
        "cv_auc_std": round(float(np.std(fold_auc)), 4),
        "cv_fold_auc": [round(a, 4) for a in fold_auc],
        "member_cv_auc": member_auc,
        "cv_folds": len(fold_auc),
        "evaluation": "patient-level 5-fold CV (OOF), mean of 3 full-region VGG16 runs, binary AD vs CN",
        "classes": [{"label": names[i], "precision": round(p[i], 2), "recall": round(r[i], 2),
                     "f1": round(f1[i], 2), "support": int(s[i])} for i in (0, 1)],
        "test_samples": int(len(y)),
        "epochs": 15,
        "roi_box_mm": {"x": [-40, 40], "y": [-45, 2], "z": [-34, 2]},
        "model": "Ensemble of 3 VGG16 (block5 fine-tuned) over all 18 axial slices of the hippocampal ROI, MNI-registered",
        "weighted_avg": {"precision": round(pw, 2), "recall": round(rw, 2), "f1": round(fw, 2)},
        "macro_avg": {"precision": round(pm, 2), "recall": round(rm, 2), "f1": round(fm, 2)},
        "confusion_matrix": {"labels": ["CN", "AD"],
                             "matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()},
        "task": "Binary classification: cognitively normal (CN) vs Alzheimer's disease (AD)",
        "preprocessing": ["Reorientation to canonical RAS", "Affine registration to MNI152 (2 mm)",
                          "Skull stripping with the MNI152 brain mask",
                          "Bilateral hippocampal ROI crop (40 x 24 x 18 voxels)",
                          "18 axial slices resized to 128 x 128, grouped as 6 RGB images"],
        "cohort": {"subjects": int(len(y)), "CN": int((y == 0).sum()), "AD": int((y == 1).sum())},
    }
    pd.DataFrame({"subject": ref.index, "y_true": y, "prob_ad": prob, "fold": fold}).to_csv(
        os.path.join(MET, "oof_ensemble_full.csv"), index=False)
    with open(os.path.join(BASE, "models", "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: metrics[k] for k in ("accuracy", "roc_auc", "cv_auc_mean", "cv_auc_std",
                                               "cv_fold_auc", "classes", "confusion_matrix")}, indent=1))


if __name__ == "__main__":
    main()
