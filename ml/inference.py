"""
inference.py
============
Inference pipeline for the deployed model: binary CN vs AD classification from
a bilateral hippocampal region of interest on MNI-registered T1 MRI.

Train/serve parity: every preprocessing step is imported from the scripts that
built the training cache (preprocess_register_roi.py and train_roi_fullregion.py),
so the server applies exactly the transformation the model was trained on.

Pipeline per uploaded scan:
  1. reorient to canonical RAS
  2. affine registration to the MNI152 2 mm template (dipy)
  3. skull stripping with the MNI152 brain mask
  4. crop the hippocampal box (MNI x -40..40, y -45..2, z -34..2 mm)
  5. all 18 axial slices of the box, resized to 128x128 (6 RGB groups of 3)
  6. average of the AD probability of three independently trained networks
"""

import os
import time
import logging
import numpy as np
import tensorflow as tf

from preprocess_register import _load_template
from preprocess_register_roi import register_and_crop, full_res_box
from train_roi_fullregion import roi_crop_to_stack

# The preprocessing scripts call logging.disable(CRITICAL) at import time to keep
# dipy quiet during batch runs. Inside the API that would silence every logger,
# so undo it and quiet only dipy's per-step INFO messages.
logging.disable(logging.NOTSET)
logging.getLogger("dipy").setLevel(logging.WARNING)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Comma-separated list; the deployed model is the mean of three ensemble members.
MODEL_PATHS = os.environ.get(
    "MODEL_PATHS",
    ",".join(os.path.join(BASE_DIR, "models", f"final_full_roi_m{i}.keras") for i in (1, 2, 3)),
).split(",")
METRICS_PATH = os.environ.get(
    "METRICS_PATH", os.path.join(BASE_DIR, "models", "final_metrics.json")
)

CLASS_CODES = ["CN", "AD"]
CLASS_NAMES = {
    0: "CN, cognitively normal",
    1: "AD, Alzheimer's disease",
}
CLASS_ANALYSIS = {
    0: ("The hippocampal region does not show the atrophy pattern the model "
        "associates with Alzheimer's disease. Routine clinical follow-up is "
        "still recommended."),
    1: ("The hippocampal region shows a pattern the model associates with "
        "Alzheimer's disease. Specialist review and clinical correlation are "
        "strongly recommended."),
}


class _CompatDense(tf.keras.layers.Dense):
    """Ignore the 'quantization_config' argument written by newer Keras."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)


class Classifier:
    """Loads the ensemble and the MNI template once, then classifies scans."""

    def __init__(self, model_paths: list[str] = MODEL_PATHS):
        self.models = [
            tf.keras.models.load_model(p, custom_objects={"Dense": _CompatDense}, compile=False)
            for p in model_paths
        ]
        self.static, self.static_aff, self.brain_mask = _load_template()
        self.box = full_res_box(self.static.shape, self.static_aff)

    def preprocess(self, nifti_path: str) -> np.ndarray:
        crop = register_and_crop(
            nifti_path, self.static, self.static_aff, self.brain_mask, self.box
        )
        n_keep = crop.shape[2] - crop.shape[2] % 3
        return roi_crop_to_stack(crop[:, :, :n_keep])[np.newaxis, ...]

    def classify(self, nifti_path: str) -> dict:
        t0 = time.time()
        x = self.preprocess(nifti_path)
        probs = np.mean([m.predict(x, verbose=0)[0] for m in self.models], axis=0)
        idx = int(np.argmax(probs))
        return {
            "prediction": CLASS_NAMES[idx],
            "class_code": CLASS_CODES[idx],
            "class_index": idx,
            "confidence": float(probs[idx]),
            "all_probabilities": [float(p) for p in probs],
            "class_labels": CLASS_CODES,
            "analysis": CLASS_ANALYSIS[idx],
            "processing_seconds": round(time.time() - t0, 1),
        }
