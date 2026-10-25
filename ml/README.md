# Backend and research pipeline

This folder holds the API, the deployed inference pipeline, and every script used during the research. The scripts are kept on purpose: together they are the experimental record behind the thesis.

## Final pipeline (what is deployed)

| Step | Script | Output |
|---|---|---|
| 1. Select one scan per patient | `train_cnn_cv.py` (`select_one_scan_per_patient`) | patient list with labels |
| 2. Register to MNI152, skull strip, crop the hippocampal box | `preprocess_register_roi.py` | `cache_reg_binary_roi/<subject>.npy`, `cache_binary_roi_hires.npz` |
| 3. Validate (5-fold, patient level), three runs | `train_roi_fullregion.py` | out-of-fold predictions and metrics |
| 4. Ensemble metrics | `build_ensemble_metrics.py` | `models/final_metrics.json` |
| 5. Train the three ensemble members on all subjects | `train_roi_fullregion.py` (`FINAL=1`) | `models/final_full_roi_m{1,2,3}.keras` |
| 6. Serve | `inference.py` + `api.py` | `/predict` (mean of the three members) |

The deployed model sees all 18 axial slices of the hippocampal box: they are grouped into 6 RGB images of 3 slices, one shared VGG16 encodes each, and the features are concatenated. Three networks trained with the same recipe are averaged.

`inference.py` imports its preprocessing directly from steps 2 and 3, so the server applies exactly the transformation the model was trained on. This was checked: the server's input for a test subject matches the training cache with correlation 1.000.

### Reproducing the final model

```powershell
..\venv\Scripts\python.exe preprocess_register_roi.py      # about 2 hours on CPU, resumable
foreach ($t in "", "_rep2", "_rep3") { $env:ROI_TAG=$t; ..\venv\Scripts\python.exe train_roi_fullregion.py }   # ~2 h each
..\venv\Scripts\python.exe build_ensemble_metrics.py
$env:FINAL="1"; foreach ($i in 1, 2, 3) { $env:ROI_TAG="_m$i"; ..\venv\Scripts\python.exe train_roi_fullregion.py }
```

The three `models/final_full_roi_m*.keras` files and `models/final_metrics.json` are what the API loads.

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
| 13 | Experiment 11 repeated 5 times | `train_roi_binary.py` (`SKIP_FINAL=1`) | AUC 0.772 to 0.817, mean 0.800 ± 0.019 |
| 14 | **All 18 slices of the region**, 3 runs | `train_roi_fullregion.py` | AUC 0.850, 0.858, 0.859 (mean 0.856) |
| 15 | **Mean of the 3 runs (deployed)** | `build_ensemble_metrics.py` | **AUC 0.868 ± 0.022, accuracy 80.3%** |

Key findings:

* **Leakage.** The original 82% came from splitting by image instead of by patient. Each patient had several preprocessed copies of the same brain, so test brains were also in training.
* **Mixed orientations.** 551 of the first 841 scans were stored as (I,P,L) and 290 as (R,A,S). Until they were reoriented, the network was comparing rotated anatomy.
* **Registration and region of interest** gave the largest gains. Changing architectures did not.
* **More data stabilized the result** without raising it, which marked the ceiling of the 3-slice approach.
* **The limit was how much of the region the model sees.** With 3 slices the CNN ties logistic regression on the same slices (0.80 vs 0.785) and loses to logistic regression on the full region (0.826). Given all 18 slices, the CNN reaches 0.856, and the mean of three runs 0.868, which beats logistic regression on the full region (paired bootstrap, p = 0.004).
* **Run-to-run variation.** TensorFlow on CPU is not deterministic: the 3-slice model varied by ±0.019 AUC between identical runs, the full-region model by ±0.005.

## Thesis figures

`make_thesis_figures.py <output_dir> [--images]` regenerates every chart and MRI figure used in the thesis from the logs, metrics and caches.

## Training caches

The `cache_*` folders and `*.npz` files hold preprocessed data (several GB). They are regenerable and excluded from git and Docker.
