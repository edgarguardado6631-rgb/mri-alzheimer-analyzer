"""
analysis_field_strength.py
==========================
Checks the acquisition confound of the final cohort. In ADNI 1 every T1 scan
used here is 1.5 T with the MPR/N3 pipeline, and in ADNI GO/2 every scan is
3 T with MT1/N3m, so field strength, phase and preprocessing are one variable.

Reports, for logistic regression (recomputed, same folds) and for the CNN
(from the saved out-of-fold predictions):
  * performance at each field strength,
  * the AUC of field strength alone as a predictor of AD (size of the confound).

Field strength of each scan comes from results/metrics/final_cohort_field.csv
(matched by ADNI series ID; the scans without a matching series are assigned by
pipeline, which agreed with field strength in every matched scan).
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

BASE = os.path.dirname(os.path.abspath(__file__))
MET = os.path.join(BASE, "results", "metrics")


def summarize(y, prob, field):
    out = {}
    for name, mask in (("todos", np.ones(len(y), bool)), ("1.5T", field == "1.5T"), ("3T", field == "3T")):
        yy, pp = y[mask], prob[mask]
        tn, fp, fn, tp = confusion_matrix(yy, (pp >= 0.5).astype(int), labels=[0, 1]).ravel()
        out[name] = {"n": int(mask.sum()), "CN": int((yy == 0).sum()), "AD": int((yy == 1).sum()),
                     "auc": round(roc_auc_score(yy, pp), 3),
                     "sens": round(tp / (tp + fn), 3), "spec": round(tn / (tn + fp), 3)}
    return out


def main():
    d = np.load(os.path.join(BASE, "cache_binary_roi_hires.npz"))
    V, y, pids = d["X"], d["y"], d["pids"]
    fld = pd.read_csv(os.path.join(MET, "final_cohort_field.csv"))
    fld["field"] = fld["field"].fillna(pd.Series(np.where(fld["mt1"], "3T", "1.5T"), index=fld.index))
    field = fld.set_index("subject").loc[pids, "field"].to_numpy()

    res = {"cohorte": {}}
    ct = pd.crosstab(pd.Series(y, name="y"), pd.Series(field, name="campo"))
    res["cohorte"]["tabla"] = {f"{'AD' if k == 1 else 'CN'}": ct.loc[k].to_dict() for k in ct.index}
    res["cohorte"]["chi2_p"] = round(float(stats.chi2_contingency(ct)[1]), 3)
    # how well would field strength alone separate AD from CN?
    res["cohorte"]["auc_solo_campo"] = round(roc_auc_score(y, (field == "1.5T").astype(float)), 3)

    X = V.reshape(len(V), -1)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    prob = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.001, max_iter=5000, class_weight="balanced"))
        prob[te] = m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    res["regresion_logistica"] = summarize(y, prob, field)

    oof = os.path.join(MET, "oof_binary_roi_oof.csv")
    if os.path.exists(oof):
        o = pd.read_csv(oof).set_index("subject").loc[pids]
        assert (o["y_true"].to_numpy() == y).all()
        res["cnn"] = summarize(y, o["prob_ad"].to_numpy(), field)

    json.dump(res, open(os.path.join(MET, "field_strength_analysis.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
