"""
check_parity.py
===============
Checks that the deployed pipeline (inference.Classifier) prepares a scan exactly
as the training cache did, and times a full prediction.

    python check_parity.py [subject ...]

For each subject it registers the raw NIfTI through the server code, compares
the model input with the one built from the training cache (Pearson r), and
prints the ensemble's AD probability and the processing time.
"""

import sys
import time
import numpy as np

import preprocess_register_roi as roi
from train_roi_fullregion import roi_crop_to_stack
from inference import Classifier


def main(subjects):
    rows = {pid: path for path, pid, _ in roi.select_one_scan_per_patient(roi.DATA_DIRS, roi.LABELS_FILE)}
    clf = Classifier()
    for pid in subjects:
        cached = roi_crop_to_stack(np.load(f"{roi.CACHE_DIR}/{pid}.npy"))
        t0 = time.time()
        served = clf.preprocess(rows[pid])[0]
        r = np.corrcoef(cached.ravel(), served.ravel())[0, 1]
        res = clf.classify(rows[pid])
        print(f"{pid}: r={r:.4f}  max|diff|={np.abs(cached - served).max():.2e}  "
              f"P(AD)={res['all_probabilities'][1]:.3f}  classify={res['processing_seconds']}s  "
              f"preprocess={time.time() - t0 - res['processing_seconds']:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:] or ["002_S_0295", "002_S_0619"])
