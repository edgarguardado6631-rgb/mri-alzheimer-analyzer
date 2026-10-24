# Backend and research pipeline

This folder holds the API, the deployed inference pipeline, and every script used during the research. The scripts are kept on purpose: together they are the experimental record behind the thesis.

## Final pipeline (what is deployed)

| Step | Script | Output |
|---|---|---|
| 1. Select one scan per patient | `train_cnn_cv.py` (`select_one_scan_per_patient`) | patient list with labels |
| 2. Register to MNI152, skull strip, crop the hippocampal box | `preprocess_register_roi.py` | `cache_reg_binary_roi/<subject>.npy`, `cache_binary_roi_hires.npz` |
| 3. Train and validate (5-fold, patient level) | `train_roi_binary.py` | metrics, figures, final model |
| 4. Serve | `inference.py` + `api.py` | `/predict` |

`inference.py` imports its preprocessing directly from steps 2 and 3, so the server applies exactly the transformation the model was trained on. This was checked: the server's input for a test subject matches the training cache with correlation 1.000.

### Reproducing the final model

```powershell
..\venv\Scripts\python.exe preprocess_register_roi.py      # about 2 hours on CPU, resumable
$env:ROI_TAG="_final"; ..\venv\Scripts\python.exe train_roi_binary.py
```

Then copy the resulting `alzheimer_cnn_binary_roi_final.h5` to `models/final_binary_roi.h5`.

### Data layout

```
data/
├── Thesis_CNMCIAD_N3_841/ADNI/<subject>/...   first download (841 subjects)
├── new/images/ADNI/<subject>/...              second download (1,744 subjects)
├── new/meta/ADNI/*.xml                        per-image ADNI metadata
└── cohort/
    ├── selected_cohort.csv     labels for the first download
    ├── merged_cohort.csv       Image ID to diagnosis for both downloads
    └── final_cohort.csv        the 824 subjects of the final model, with sex and age
```

## Research history

Every number below comes from patient-level 5-fold cross-validation. Logs are in `results/logs/`, metrics in `results/metrics/`.

| # | Experiment | Script | Result |
|---|---|---|---|
| 0 | Original single split (data leakage) | `train_cnn.py` | 82% accuracy, **invalid** |
| 1 | Single central slice, 3 classes, 841 patients | `train_cnn_cv.py` | 50.9% accuracy |
| 2 | Three slices, 3 classes | `train_cnn_cv.py` | 49.5% |
| 3 | 3D CNN on the whole volume, 3 classes | `train_cnn_3d.py` | 48.5% |
| 4 | Binary CN vs AD, single slice | `train_cnn_binary.py` | AUC 0.56 |
| 5 | Reorientation to RAS | `train_cnn_binary.py` | AUC 0.65 |
| 6 | MNI registration and skull stripping | `preprocess_register.py` | AUC 0.69 |
| 7 | 3D CNN on registered volumes | `train_3d_binary.py` | AUC 0.58 |
| 8 | Hippocampal region of interest | `train_roi_binary.py` | AUC 0.79 |
| 9 | Left/right flip augmentation | `train_roi_binary.py` (`ROI_FLIP=1`) | AUC 0.78, rejected |
| 10 | Full resolution ROI | `preprocess_register_roi.py` | AUC 0.80 |
| 11 | **Second download, 824 subjects** | same | **AUC 0.80, accuracy 72.8%** |
| 12 | Classical baselines, same folds | `baseline_classical.py` | LR 0.826, linear SVM 0.814, RF 0.799, RBF SVM 0.789; LR on the CNN's 3 slices 0.785 |

Key findings:

* **Leakage.** The original 82% came from splitting by image instead of by patient. Each patient had several preprocessed copies of the same brain, so test brains were also in training.
* **Mixed orientations.** 551 of the first 841 scans were stored as (I,P,L) and 290 as (R,A,S). Until they were reoriented, the network was comparing rotated anatomy.
* **Registration and region of interest** gave the largest gains. Changing architectures did not.
* **More data stabilized the result** without raising it, which marks the ceiling of this 2D approach.
* **Classical models are competitive.** Logistic regression on the full 3D region (18 slices) beats the CNN (3 slices), but with the same 3 slices the CNN wins. The limit is how much of the region the model sees.

## Thesis figures

`make_thesis_figures.py <output_dir> [--images]` regenerates every chart and MRI figure used in the thesis from the logs, metrics and caches.

## Training caches

The `cache_*` folders and `*.npz` files hold preprocessed data (several GB). They are regenerable and excluded from git and Docker.
