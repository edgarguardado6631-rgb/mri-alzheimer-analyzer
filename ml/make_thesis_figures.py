"""
make_thesis_figures.py
======================
Regenerates the figures used in the thesis from the experiment logs, metrics
and cached data, so every figure can be traced back to code.

    python make_thesis_figures.py <output_dir> [--images]

--images also renders the MRI figures (orientation, registration, ROI inputs);
these need the ADNI data and take about a minute (one registration).
Palette: blue CN / orange AD (validated for color-vision deficiency),
gray for reference or rejected results.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

BLUE, ORANGE, GRAY, LIGHT_GRAY = "#2a78d6", "#eb6834", "#8a8984", "#d9d8d4"
INK, INK_2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def _save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("saved", path)


# ── 1. Experiment progression ───────────────────────────────────────────────
def fig_progression(out):
    three = [  # (label, accuracy, kept)
        ("División por imagen (fuga de datos)", 0.82, None),
        ("Primer conjunto, 179 pacientes", 0.408, True),
        ("841 pacientes, un corte axial", 0.509, True),
        ("841 pacientes, tres cortes", 0.495, False),
        ("CNN 3D, volumen completo", 0.485, False),
        ("CNN 3D, volumen reorientado", 0.419, False),
    ]
    binary = [  # (label, AUC, kept)
        ("Un corte, sin alinear", 0.579, True),
        ("+ reorientación RAS", 0.655, True),
        ("+ registro MNI y sin cráneo", 0.695, True),
        ("CNN 3D sobre volumen registrado", 0.585, False),
        ("+ región del hipocampo", 0.787, True),
        ("+ volteo izquierda/derecha", 0.778, False),
        ("+ región a resolución completa", 0.798, True),
        ("+ segunda descarga (824 sujetos)", 0.804, True),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"wspace": 0.95})

    labels = [t[0] for t in three][::-1]
    vals = [t[1] for t in three][::-1]
    kept = [t[2] for t in three][::-1]
    colors = [LIGHT_GRAY if k is None else (BLUE if k else GRAY) for k in kept]
    bars = a.barh(labels, vals, color=colors, height=0.62, edgecolor="white", linewidth=2)
    bars[-1].set_hatch("///")
    bars[-1].set_edgecolor(GRAY)
    for bar, v, k in zip(bars, vals, kept):
        txt = f"{v * 100:.0f}%" + ("  (inválido)" if k is None else "")
        a.text(v + 0.012, bar.get_y() + bar.get_height() / 2, txt, va="center", color=INK, fontsize=9)
    a.axvline(1 / 3, color=INK_2, ls="--", lw=1)
    a.set_ylim(-0.6, len(labels) + 0.1)
    a.text(1 / 3 + 0.01, len(labels) - 0.3, "azar (33%)", color=INK_2, fontsize=8, ha="left")
    a.set_xlim(0, 1.05)
    a.set_xlabel("Exactitud (validación cruzada por paciente)")
    a.set_title("A. Tres clases: CN, MCI y AD", loc="left", fontsize=11, color=INK)

    labels = [t[0] for t in binary][::-1]
    vals = [t[1] for t in binary][::-1]
    kept = [t[2] for t in binary][::-1]
    bars = b.barh(labels, vals, color=[BLUE if k else GRAY for k in kept], height=0.62,
                  edgecolor="white", linewidth=2)
    for bar, v in zip(bars, vals):
        b.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", color=INK, fontsize=9)
    b.axvline(0.5, color=INK_2, ls="--", lw=1)
    b.set_ylim(-0.6, len(labels) + 0.1)
    b.text(0.51, len(labels) - 0.3, "azar (0.50)", color=INK_2, fontsize=8, ha="left")
    b.set_xlim(0, 1.0)
    b.set_xlabel("AUC ROC (validación cruzada por paciente)")
    b.set_title("B. Dos clases: CN contra AD", loc="left", fontsize=11, color=INK)

    for ax in (a, b):
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    handles = [Rectangle((0, 0), 1, 1, color=BLUE), Rectangle((0, 0), 1, 1, color=GRAY)]
    fig.legend(handles, ["Cambio conservado", "Cambio descartado"], loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    _save(fig, out, "fig_progresion.png")


# ── 2. Confusion matrix of the final model ──────────────────────────────────
def fig_confusion(out):
    m = json.load(open(os.path.join(BASE_DIR, "models", "final_metrics.json")))
    cm = np.array(m["confusion_matrix"]["matrix"])
    names = ["CN", "AD"]
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / cm[i].sum() * 100
            color = "white" if pct > 55 else INK
            ax.text(j, i, f"{cm[i, j]}\n({pct:.0f}%)", ha="center", va="center", color=color, fontsize=12)
    ax.set_xticks([0, 1], [f"Predicho {n}" for n in names])
    ax.set_yticks([0, 1], [f"Real {n}" for n in names])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Predicciones fuera de pliegue, 824 sujetos", fontsize=10, color=INK_2)
    _save(fig, out, "fig_matriz_confusion.png")


def fig_roc(out):
    """Out-of-fold ROC of the final model. The per-subject probabilities were
    not stored during training, so the curve was digitized from the plot the
    training run produced (results/figures/roc_curve_binary_roi_merged824.png);
    the digitized area is 0.803 against the reported 0.800."""
    path = os.path.join(BASE_DIR, "results", "metrics", "roc_final_digitized.csv")
    if not os.path.exists(path):
        return
    roc = pd.read_csv(path)
    m = json.load(open(os.path.join(BASE_DIR, "models", "final_metrics.json")))
    sens = m["classes"][1]["recall"]
    spec = m["classes"][0]["recall"]
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot(roc.fpr, roc.tpr, color=BLUE, lw=2, label=f"CNN, AUC = {m['roc_auc']:.2f}")
    ax.plot([0, 1], [0, 1], ls="--", color=GRAY, lw=1, label="Azar, AUC = 0.50")
    ax.scatter([1 - spec], [sens], s=60, color=ORANGE, zorder=3, edgecolor="white", linewidth=2,
               label=f"Umbral 0.5: sens. {sens:.2f}, espec. {spec:.2f}")
    ax.set_xlabel("1 menos especificidad (falsos positivos)")
    ax.set_ylabel("Sensibilidad (verdaderos positivos)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _save(fig, out, "fig_roc.png")


# ── 3. Cohort demographics ──────────────────────────────────────────────────
def fig_demographics(out, mmse_csv=None):
    df = pd.read_csv(os.path.join(ROOT, "data", "cohort", "final_cohort.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"wspace": 0.35})
    a, b, c = axes
    bins = np.arange(55, 95, 2.5)
    for grp, col in (("CN", BLUE), ("AD", ORANGE)):
        ages = df[df["Research Group"] == grp]["Age"].dropna()
        a.hist(ages, bins=bins, histtype="step", lw=2, color=col,
               label=f"{grp} (n={len(ages)}, {ages.mean():.1f} ± {ages.std():.1f})")
    a.set_ylim(0, a.get_ylim()[1] * 1.4)
    a.set_xlabel("Edad (años)")
    a.set_ylabel("Sujetos")
    a.legend(frameon=False, fontsize=8, loc="upper left")
    a.set_title("A. Edad", loc="left", color=INK)

    if mmse_csv and os.path.exists(mmse_csv):
        mm = pd.read_csv(mmse_csv)
        data = [mm[mm.grp == g].mmse.dropna() for g in ("CN", "AD")]
        parts = b.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                          medianprops={"color": INK})
        for patch, col in zip(parts["boxes"], (BLUE, ORANGE)):
            patch.set_facecolor(col)
            patch.set_alpha(0.35)
            patch.set_edgecolor(col)
        rng = np.random.default_rng(0)
        for i, (d, col) in enumerate(zip(data, (BLUE, ORANGE)), start=1):
            b.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=6, color=col, alpha=0.5, lw=0)
        b.set_xticks([1, 2], [f"CN (n={len(data[0])})", f"AD (n={len(data[1])})"])
        b.set_ylabel("Puntaje MMSE")
    b.set_title("B. Estado cognitivo (MMSE)", loc="left", color=INK)

    sex = df.groupby(["Research Group", "Sex"]).size().unstack()
    x = np.arange(2)
    for k, (s, lab) in enumerate((("F", "Mujeres"), ("M", "Hombres"))):
        vals = [sex.loc["CN", s], sex.loc["AD", s]]
        bars = c.bar(x + (k - 0.5) * 0.36, vals, width=0.34,
                     color=[BLUE, ORANGE], alpha=0.55 if s == "F" else 1.0,
                     edgecolor="white", linewidth=2)
        for bar, v in zip(bars, vals):
            c.text(bar.get_x() + bar.get_width() / 2, v + 4, f"{lab[0]} {v}", ha="center", fontsize=8, color=INK)
    c.set_xticks(x, ["CN", "AD"])
    c.set_ylabel("Sujetos")
    c.set_title("C. Sexo (M = mujeres, H = hombres)", loc="left", color=INK)
    _save(fig, out, "fig_demografia.png")


# ── 4. CNN vs classical baselines ───────────────────────────────────────────
def fig_model_comparison(out):
    path = os.path.join(BASE_DIR, "results", "metrics", "baseline_classical.json")
    if not os.path.exists(path):
        print("skip comparison: run baseline_classical.py first")
        return
    base = json.load(open(path, encoding="utf-8"))
    m = json.load(open(os.path.join(BASE_DIR, "models", "final_metrics.json")))
    rows = [(k.replace("(vóxeles)", "(región 3D)").replace("(PCA 100)", "(región 3D, PCA)"),
             v["cv_auc_mean"], v["cv_auc_std"], False) for k, v in base.items()]
    lr3 = os.path.join(BASE_DIR, "results", "metrics", "baseline_lr_3slices.json")
    if os.path.exists(lr3):
        v = json.load(open(lr3))
        rows.append(("Regresión logística (3 cortes)", v["cv_auc_mean"], v["cv_auc_std"], False))
    rows.append(("CNN VGG16 (3 cortes, este trabajo)", m["cv_auc_mean"], m["cv_auc_std"], True))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], xerr=[r[2] for r in rows], height=0.6,
            color=[BLUE if r[3] else GRAY for r in rows], edgecolor="white", linewidth=2,
            error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
    for yi, r in zip(y, rows):
        ax.text(r[1] + r[2] + 0.012, yi, f"{r[1]:.2f} ± {r[2]:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.axvline(0.5, color=INK_2, ls="--", lw=1)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("AUC ROC (media ± desviación entre los 5 pliegues)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    _save(fig, out, "fig_comparacion_modelos.png")


# ── 5-7. MRI figures ────────────────────────────────────────────────────────
def _show(ax, img, title):
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_title(title, fontsize=9, color=INK)
    ax.axis("off")


def _window(sl):
    nz = sl[sl > 0]
    lo, hi = (np.percentile(nz, [1, 99]) if nz.size else (sl.min(), sl.max()))
    return np.clip((sl - lo) / (hi - lo + 1e-8), 0, 1)


def fig_orientation(out, ipl_path, ras_path):
    import nibabel as nib
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.6))
    for col, (path, code) in enumerate(((ipl_path, "I,P,L"), (ras_path, "R,A,S"))):
        img = nib.load(path)
        data = img.get_fdata()
        ax_codes = nib.aff2axcodes(img.affine)
        si = next(i for i, c in enumerate(ax_codes) if c in ("S", "I"))
        mid = data.shape[si] // 2
        naive = np.take(data, mid, axis=si)          # what the first pipeline fed the CNN
        _show(axes[0, col], _window(naive), f"Antes: almacenado como ({code})")
        can = nib.as_closest_canonical(img).get_fdata()
        _show(axes[1, col], _window(np.rot90(can[:, :, can.shape[2] // 2 - 8])), "Después: reorientado a RAS")
    fig.tight_layout()
    _save(fig, out, "fig_orientacion.png")


def fig_registration(out, nifti_path):
    import nibabel as nib
    logging.disable(logging.CRITICAL)
    from dipy.align import affine_registration
    from preprocess_register import (_load_template, REG_PIPELINE, REG_LEVEL_ITERS,
                                     REG_SIGMAS, REG_FACTORS, REG_NBINS, REG_SAMPLING)
    from preprocess_register_roi import full_res_box
    static, aff, mask = _load_template()
    (x0, y0, z0), (x1, y1, z1) = full_res_box(static.shape, aff)
    img = nib.as_closest_canonical(nib.load(nifti_path))
    mov = img.get_fdata().astype("float32")
    xf, _ = affine_registration(mov, static, moving_affine=img.affine, static_affine=aff,
                                pipeline=REG_PIPELINE, level_iters=REG_LEVEL_ITERS, sigmas=REG_SIGMAS,
                                factors=REG_FACTORS, nbins=REG_NBINS, sampling_proportion=REG_SAMPLING)
    stripped = xf * mask
    zc, yc = (z0 + z1) // 2, (y0 + y1) // 2
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.3))
    _show(axes[0], _window(np.rot90(mov[:, :, mov.shape[2] // 2 - 8])), "1. Original (RAS)")
    _show(axes[1], _window(np.rot90(static[:, :, zc])), "2. Plantilla MNI152")
    _show(axes[2], _window(np.rot90(xf[:, :, zc])), "3. Registrado al MNI152")
    _show(axes[3], _window(np.rot90(stripped[:, yc, :])), "4. Sin cráneo, región del hipocampo")
    Y, Z = static.shape[1], static.shape[2]
    axes[2].add_patch(Rectangle((x0, Y - y1), x1 - x0, y1 - y0, fill=False, ec=ORANGE, lw=2))
    axes[3].add_patch(Rectangle((x0, Z - z1), x1 - x0, z1 - z0, fill=False, ec=ORANGE, lw=2))
    fig.tight_layout()
    _save(fig, out, "fig_registro.png")


def fig_roi_inputs(out):
    from train_roi_binary import roi_crop_to_image
    df = pd.read_csv(os.path.join(ROOT, "data", "cohort", "final_cohort.csv"))
    picks = []
    for grp in ("CN", "AD"):
        sids = df[df["Research Group"] == grp]["Subject ID"].tolist()
        picks += [(grp, s) for s in sids if os.path.exists(os.path.join(BASE_DIR, "cache_reg_binary_roi", f"{s}.npy"))][:2]
    fig, axes = plt.subplots(len(picks), 3, figsize=(6.2, 2.1 * len(picks)))
    for r, (grp, sid) in enumerate(picks):
        img = roi_crop_to_image(np.load(os.path.join(BASE_DIR, "cache_reg_binary_roi", f"{sid}.npy")))
        for c, name in enumerate(("inferior", "medio", "superior")):
            _show(axes[r, c], np.rot90(img[:, :, c]), f"{grp} {sid}, corte {name}" if c == 1 else f"corte {name}")
    fig.tight_layout()
    _save(fig, out, "fig_roi_entradas.png")


if __name__ == "__main__":
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    fig_progression(out_dir)
    fig_confusion(out_dir)
    fig_roc(out_dir)
    fig_demographics(out_dir, os.path.join(BASE_DIR, "results", "metrics", "final_cohort_mmse.csv"))
    fig_model_comparison(out_dir)
    if "--images" in sys.argv:
        adni = os.path.join(ROOT, "data", "Thesis_CNMCIAD_N3_841", "ADNI")
        ipl = os.path.join(adni, "002_S_0295", "MPR__GradWarp__B1_Correction__N3", "2006-04-18_08_20_30.0", "I45107",
                           "ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3_Br_20070319113435616_S13408_I45107.nii")
        ras = os.path.join(adni, "002_S_4213", "MT1__N3m", "2011-09-02_18_40_57.0", "I255409",
                           "ADNI_002_S_4213_MR_MT1__N3m_Br_20110910135704514_S121168_I255409.nii")
        fig_orientation(out_dir, ipl, ras)
        fig_registration(out_dir, ipl)
        fig_roi_inputs(out_dir)
