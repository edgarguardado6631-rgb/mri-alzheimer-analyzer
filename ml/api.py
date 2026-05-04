from dotenv import load_dotenv
import pathlib
load_dotenv(pathlib.Path(__file__).parent / '.env', override=True)

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
import numpy as np
import nibabel as nib
import os
import re
import glob
import io
import logging
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI()

# ── Rate limiter ───────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=[])
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Too many requests. Please slow down."})

# ── Security headers middleware ────────────────────────────────────────────
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# ── CORS ───────────────────────────────────────────────────────────────────
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in _raw_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,          # no cookies needed
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# ── Constants ──────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "alzheimer_cnn_model.h5")
MAX_UPLOAD_MB = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Allowlist: patient_id must be alphanumeric + hyphens/underscores only
PATIENT_ID_RE = re.compile(r'^[A-Za-z0-9_\-]+$')

model = None


# ── Input validators ───────────────────────────────────────────────────────
def validate_patient_id(patient_id: str) -> str:
    if not PATIENT_ID_RE.match(patient_id):
        raise HTTPException(status_code=400, detail="Invalid patient ID format.")
    return patient_id


def validate_scan_filename(filename: str) -> str:
    """Block path traversal and enforce NIfTI extension."""
    if '..' in filename or filename.startswith('/') or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if not (filename.endswith('.nii') or filename.endswith('.nii.gz')):
        raise HTTPException(status_code=400, detail="Only .nii / .nii.gz files are supported.")
    return filename


def safe_data_path(base: str, *parts: str) -> str:
    """Resolve a path and ensure it stays within base directory."""
    resolved = os.path.normpath(os.path.join(base, *parts))
    if not resolved.startswith(os.path.normpath(base)):
        raise HTTPException(status_code=403, detail="Access denied.")
    return resolved


# ── Keras version compatibility shim ──────────────────────────────────────
class _CompatDense(tf.keras.layers.Dense):
    """Strips unknown 'quantization_config' added by newer Keras versions."""
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)


@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={'Dense': _CompatDense},
                compile=False,
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    else:
        logger.warning("Model file not found. Predictions will be mocked until training is run.")


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "Alzheimer's MRI Analysis API"}


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    # Validate extension
    if not (file.filename.endswith(".nii") or file.filename.endswith(".nii.gz")):
        raise HTTPException(status_code=400, detail="Only .nii or .nii.gz files are supported.")

    # Read with size cap
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_UPLOAD_MB} MB.")

    # Write to isolated temp file (no user-controlled path component)
    suffix = '.nii.gz' if file.filename.endswith('.nii.gz') else '.nii'
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(content)
        tmp.flush()
        tmp.close()
        temp_path = tmp.name

        LABEL_MAP = {
            0: "CN — Cognitively Normal",
            1: "MCI — Mild Cognitive Impairment",
            2: "AD — Alzheimer's Disease",
        }
        ANALYSIS_MAP = {
            0: "The scan shows no significant patterns associated with Alzheimer's disease. Continued routine monitoring is recommended.",
            1: "The scan shows patterns that may be consistent with Mild Cognitive Impairment. Clinical follow-up and longitudinal monitoring are advised.",
            2: "The scan shows patterns consistent with Alzheimer's Disease. Further clinical correlation and specialist review are strongly recommended.",
        }

        # Validate it's actually a NIfTI file before feeding to the model
        try:
            img = nib.load(temp_path)
            data = img.get_fdata()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid NIfTI file. Please upload a valid .nii or .nii.gz scan.")

        # Axis-aware slice selection
        axcodes = nib.aff2axcodes(img.affine)
        si_axis = next((idx for idx, c in enumerate(axcodes) if c in ['S', 'I']), 2)
        mid_slice = data.shape[si_axis] // 2
        if si_axis == 0:
            slice_img = data[mid_slice, :, :]
        elif si_axis == 1:
            slice_img = data[:, mid_slice, :]
        else:
            slice_img = data[:, :, mid_slice]

        slice_img = tf.image.resize(slice_img[..., np.newaxis], (128, 128)).numpy()
        vmin, vmax = slice_img.min(), slice_img.max()
        slice_img = (slice_img - vmin) / (vmax - vmin + 1e-8)
        input_data = np.array([slice_img])

        if model:
            prediction = model.predict(input_data)
            class_idx  = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][class_idx])
            label      = LABEL_MAP.get(class_idx, f"Unknown class {class_idx}")
            analysis   = ANALYSIS_MAP.get(class_idx, "Further clinical correlation recommended.")
            all_probs  = prediction[0].tolist()
        else:
            import random
            class_idx  = random.randint(0, 2)
            confidence = random.uniform(0.7, 0.99)
            label      = f"MOCK — {LABEL_MAP[class_idx]}"
            analysis   = "This is a mock result. Train the model to get real predictions."
            all_probs  = []

        return {
            "prediction":       label,
            "confidence":       confidence,
            "analysis":         analysis,
            "class_index":      class_idx,
            "all_probabilities": all_probs,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again.")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


@app.get("/data/stats")
def get_stats():
    try:
        base_path = "data"
        adni_path = os.path.join(base_path, "Test 2", "ADNI")

        total_patients = 0
        total_scans    = 0

        if os.path.exists(adni_path):
            patients_dirs = [d for d in os.listdir(adni_path) if os.path.isdir(os.path.join(adni_path, d))]
            total_patients = len(patients_dirs)
            nii_files   = glob.glob(os.path.join(adni_path, "**", "*.nii"),    recursive=True)
            nii_gz_files = glob.glob(os.path.join(adni_path, "**", "*.nii.gz"), recursive=True)
            total_scans = len(nii_files) + len(nii_gz_files)

        return {
            "total_patients":  total_patients,
            "scans_processed": total_scans,
            "model_accuracy":  0.82,
        }
    except Exception as e:
        logger.error(f"Stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve stats.")


@app.get("/data/model-metrics")
def get_model_metrics():
    import json
    metrics_path = os.path.join(BASE_DIR, "model_metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Model metrics not found. Run training first.")
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Metrics read error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not read model metrics.")


@app.get("/data/demographics")
def get_demographics():
    try:
        base_path = "data"
        csv_files = glob.glob(os.path.join(base_path, "*.csv"))
        if not csv_files:
            return {"groups": [], "sex": [], "age_bins": []}

        import csv
        from collections import defaultdict

        group_counts: dict = defaultdict(int)
        sex_counts:   dict = defaultdict(int)
        age_values:   list = []
        seen_subjects: set = set()

        with open(csv_files[0], newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                subject = row.get("Subject", "").strip()
                if subject in seen_subjects:
                    continue
                seen_subjects.add(subject)
                group   = row.get("Group", "").strip()
                sex     = row.get("Sex",   "").strip()
                age_str = row.get("Age",   "").strip()
                if group:
                    group_counts[group] += 1
                if sex:
                    sex_counts[sex] += 1
                if age_str.isdigit():
                    age_values.append(int(age_str))

        age_bins: dict = defaultdict(int)
        for age in age_values:
            bin_start = (age // 10) * 10
            age_bins[f"{bin_start}–{bin_start + 9}"] += 1

        return {
            "groups":   [{"label": k, "count": v} for k, v in sorted(group_counts.items())],
            "sex":      [{"label": k, "count": v} for k, v in sorted(sex_counts.items())],
            "age_bins": [{"label": k, "count": v} for k, v in sorted(age_bins.items())],
        }
    except Exception as e:
        logger.error(f"Demographics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve demographics.")


@app.get("/data/patients")
def get_patients():
    try:
        base_path = "data"
        adni_path = os.path.join(base_path, "Test 2", "ADNI")
        if os.path.exists(adni_path):
            patients = [d for d in os.listdir(adni_path) if os.path.isdir(os.path.join(adni_path, d))]
            return {"patients": sorted(patients)}

        files = (
            glob.glob(os.path.join(base_path, "**", "*.nii"),    recursive=True) +
            glob.glob(os.path.join(base_path, "**", "*.nii.gz"), recursive=True)
        )
        patients = set()
        for f in files:
            parts = f.split(os.sep)
            for part in parts:
                if "_" in part and len(part.split("_")) == 3:
                    patients.add(part)
        return {"patients": sorted(list(patients))}
    except Exception as e:
        logger.error(f"Patients list error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve patient list.")


@app.get("/data/image/{patient_id}/scans")
def get_patient_scans(patient_id: str):
    patient_id = validate_patient_id(patient_id)
    try:
        base_path = "data"
        adni_path = os.path.join(base_path, "Test 2", "ADNI")
        patient_dir = safe_data_path(adni_path, patient_id)

        if os.path.exists(patient_dir):
            files = (
                glob.glob(os.path.join(patient_dir, "**", "*.nii"),    recursive=True) +
                glob.glob(os.path.join(patient_dir, "**", "*.nii.gz"), recursive=True)
            )
        else:
            files = (
                glob.glob(os.path.join(base_path, "**", f"*{patient_id}*.nii"),    recursive=True) +
                glob.glob(os.path.join(base_path, "**", f"*{patient_id}*.nii.gz"), recursive=True)
            )

        scans = [os.path.basename(f) for f in files]
        return {"scans": sorted(list(set(scans)))}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scans list error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve scan list.")


@app.get("/data/image/{patient_id}/{scan_filename}/metadata")
def get_image_metadata(patient_id: str, scan_filename: str):
    patient_id    = validate_patient_id(patient_id)
    scan_filename = validate_scan_filename(scan_filename)
    try:
        base_path = "data"
        adni_path = os.path.join(base_path, "Test 2", "ADNI", patient_id)
        pattern   = safe_data_path(adni_path, "**", scan_filename) if os.path.exists(adni_path) \
                    else os.path.join(base_path, "**", scan_filename)

        files = glob.glob(pattern, recursive=True)
        if not files:
            raise HTTPException(status_code=404, detail="Scan not found.")

        img = nib.load(files[0])
        return {
            "shape":     img.shape,
            "min_slice": 0,
            "max_slice": img.shape[2] - 1,
            "file":      os.path.basename(files[0]),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metadata error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not read scan metadata.")


@app.get("/data/image/{patient_id}/{scan_filename}/slice/{slice_idx}")
def get_image_slice(patient_id: str, scan_filename: str, slice_idx: int):
    patient_id    = validate_patient_id(patient_id)
    scan_filename = validate_scan_filename(scan_filename)

    if slice_idx < 0:
        raise HTTPException(status_code=400, detail="Slice index must be non-negative.")

    try:
        base_path = "data"
        adni_path = os.path.join(base_path, "Test 2", "ADNI", patient_id)
        pattern   = safe_data_path(adni_path, "**", scan_filename) if os.path.exists(adni_path) \
                    else os.path.join(base_path, "**", scan_filename)

        files = glob.glob(pattern, recursive=True)
        if not files:
            raise HTTPException(status_code=404, detail="Scan not found.")

        img  = nib.load(files[0])
        data = img.get_fdata()

        # Clamp to valid range
        slice_idx = min(slice_idx, data.shape[2] - 1)
        slice_img = np.rot90(data[:, :, slice_idx])

        fig = plt.figure(figsize=(4, 4))
        try:
            plt.imshow(slice_img, cmap='gray')
            plt.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            return Response(content=buf.getvalue(), media_type="image/png")
        finally:
            plt.close(fig)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Slice render error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not render scan slice.")


# ── Claude AI chat ─────────────────────────────────────────────────────────
CLINICAL_SYSTEM_PROMPT = """You are NeuroScan AI, a clinical decision-support assistant specialising in Alzheimer's disease MRI analysis.

You help clinicians interpret CNN model predictions from structural MRI scans processed with the ADNI dataset pipeline. The model classifies scans into three categories:
- CN (Cognitively Normal): No significant patterns associated with cognitive decline
- MCI (Mild Cognitive Impairment): Early-stage changes; elevated risk for progression to AD
- AD (Alzheimer's Disease): Patterns consistent with Alzheimer's disease

When a scan result is provided to you, structure your response as follows:
1. Briefly confirm the classification and confidence level in plain clinical language
2. Explain what the result implies for this patient category
3. Suggest relevant next clinical steps or monitoring considerations
4. Note any caveats (confidence threshold, model limitations, need for longitudinal data)

For general questions, answer concisely about MRI analysis, the detection pipeline, or Alzheimer's disease context.

Always be evidence-based and remind the user that AI predictions are decision-support tools — final diagnosis must be made by a qualified clinician. Keep responses clear and appropriately concise."""


class ChatMessage(BaseModel):
    role: str
    content: str


class ScanContext(BaseModel):
    filename: str
    prediction: str
    confidence: float
    class_index: int
    all_probabilities: List[float]


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    scan_context: Optional[ScanContext] = None


@app.post("/ai/chat")
@limiter.limit("20/hour")
async def ai_chat(request: Request, req: ChatRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not configured on the server.")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        system = CLINICAL_SYSTEM_PROMPT
        if req.scan_context:
            ctx = req.scan_context
            label_map = {0: "CN — Cognitively Normal", 1: "MCI — Mild Cognitive Impairment", 2: "AD — Alzheimer's Disease"}
            probs = ctx.all_probabilities
            prob_lines = "\n".join(
                f"  - {label_map.get(i, f'Class {i}')}: {p * 100:.1f}%"
                for i, p in enumerate(probs)
            ) if probs else "  Not available"
            system += f"""

Current scan under review:
  File: {ctx.filename}
  CNN prediction: {ctx.prediction}
  Confidence: {ctx.confidence * 100:.1f}%
  All class probabilities:
{prob_lines}

Interpret this result in your response."""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            messages=[{"role": m.role, "content": m.content} for m in req.messages],
        )
        return {"content": response.content[0].text}

    except anthropic.APIError as e:
        logger.error(f"Claude API error: {e}")
        raise HTTPException(status_code=502, detail="AI service unavailable. Please try again.")
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat request failed. Please try again.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
