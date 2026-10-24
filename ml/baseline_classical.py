"""
baseline_classical.py
=====================
Classical machine learning baselines for thesis Objective 5, evaluated on the
same data and the same folds as the final CNN.

Input: the full-resolution hippocampal ROI of every subject
(cache_binary_roi_hires.npz, 824 subjects, 40x24x18 voxels). Each ROI is
flattened into a voxel vector, the same "voxel-based" representation used by
the SVM baselines in Wen et al. (2020).

Folds: StratifiedKFold(5, shuffle=True, random_state=42) over the cache order,
identical to train_roi_binary.py, so every model sees exactly the same
train/test patients as the CNN. Scaling and PCA are fitted inside each training
fold (sklearn Pipeline) so no information leaks from the test fold.
"""

import os
import json
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42

MODELS = {
    "Regresión logística (vóxeles)": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=0.001, max_iter=5000, class_weight="balanced")),
    "SVM lineal (vóxeles)": lambda: make_pipeline(
        StandardScaler(), SVC(kernel="linear", C=1e-4, class_weight="balanced", random_state=RANDOM_STATE)),
    "SVM RBF (PCA 100)": lambda: make_pipeline(
        StandardScaler(), PCA(n_components=100, random_state=RANDOM_STATE),
        SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=RANDOM_STATE)),
    "Random Forest (vóxeles)": lambda: RandomForestClassifier(
        n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
}


def main():
    d = np.load(os.path.join(BASE_DIR, "cache_binary_roi_hires.npz"))
    X = d["X"].reshape(len(d["X"]), -1).astype("float32")
    y = d["y"]
    print(f"X={X.shape}, CN={int((y == 0).sum())}, AD={int((y == 1).sum())}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, build in MODELS.items():
        t0 = time.time()
        aucs, accs = [], []
        oof_pred = np.zeros(len(y), dtype=int)
        for tr, te in skf.split(X, y):
            model = build().fit(X[tr], y[tr])
            # AUC only needs a ranking score: probabilities where available,
            # the SVM decision function otherwise (no costly Platt calibration).
            if hasattr(model, "predict_proba"):
                score = model.predict_proba(X[te])[:, 1]
            else:
                score = model.decision_function(X[te])
            pred = model.predict(X[te])
            aucs.append(roc_auc_score(y[te], score))
            accs.append(accuracy_score(y[te], pred))
            oof_pred[te] = pred
        tn, fp, fn, tp = confusion_matrix(y, oof_pred, labels=[0, 1]).ravel()
        results[name] = {
            "cv_auc_mean": round(float(np.mean(aucs)), 4),
            "cv_auc_std": round(float(np.std(aucs)), 4),
            "cv_accuracy_mean": round(float(np.mean(accs)), 4),
            "cv_accuracy_std": round(float(np.std(accs)), 4),
            "sensitivity_AD": round(tp / (tp + fn), 4),
            "specificity_CN": round(tn / (tn + fp), 4),
            "per_fold_auc": [round(float(a), 4) for a in aucs],
            "seconds": round(time.time() - t0, 1),
        }
        r = results[name]
        print(f"{name:32s} AUC {r['cv_auc_mean']:.3f} ± {r['cv_auc_std']:.3f} | "
              f"acc {r['cv_accuracy_mean']:.3f} | sens {r['sensitivity_AD']:.2f} | "
              f"spec {r['specificity_CN']:.2f} | {r['seconds']}s", flush=True)

    out = os.path.join(BASE_DIR, "results", "metrics", "baseline_classical.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Saved", out)


if __name__ == "__main__":
    main()
