from dotenv import load_dotenv
import pathlib
load_dotenv(pathlib.Path(__file__).parent / ".env", override=True)

import os
import re
import io
import json
import glob
import time
import uuid
import logging
import tempfile
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import List, Optional

import numpy as np
import nibabel as nib
import pandas as pd
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import anthropic

from inference import Classifier, CLASS_NAMES, METRICS_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neuroscan")

# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.environ.get("DATA_ROOT", os.path.join(BASE_DIR, "..", "data")))

# Folders holding ADNI subject trees (ADNI/<subject>/<series>/<date>/<image>/*.nii).
# Override with IMAGE_DIRS, separated by os.pathsep.
_default_image_dirs = [
    os.path.join(DATA_ROOT, "Thesis_CNMCIAD_N3_841", "ADNI"),
    os.path.join(DATA_ROOT, "new", "images", "ADNI"),
]
IMAGE_DIRS = [
    os.path.abspath(d)
    for d in (os.environ["IMAGE_DIRS"].split(os.pathsep) if os.environ.get("IMAGE_DIRS") else _default_image_dirs)
]
COHORT_CSV = os.path.join(DATA_ROOT, "cohort", "final_cohort.csv")

MAX_UPLOAD_MB = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
PATIENT_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

CHAT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
CHAT_MAX_MESSAGES = 20          # history sent to Claude per request
CHAT_MAX_CHARS = 4000           # per message

classifier: Optional[Classifier] = None
_inference_lock = threading.Lock()     # registration is CPU heavy: one scan at a time

VIEWER_TEMP_DIR = tempfile.mkdtemp(prefix="neuroscan_viewer_")
_viewer_sessions: dict = {}            # session_id -> {"path": str, "created_at": float}
_SESSION_TTL_SECONDS = 3600


def _cleanup_viewer_sessions() -> None:
    """Remove viewer uploads older than the session TTL."""
    while True:
        time.sleep(300)
        now = time.time()
        expired = [sid for sid, info in list(_viewer_sessions.items())
                   if now - info["created_at"] > _SESSION_TTL_SECONDS]
        for sid in expired:
            path = _viewer_sessions.pop(sid, {}).get("path", "")
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        if expired:
            logger.info("Viewer cleanup removed %d expired session(s).", len(expired))


def _load_metrics() -> Optional[dict]:
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global classifier
    try:
        classifier = Classifier()
        logger.info("Model and MNI template loaded.")
    except Exception as e:
        logger.error("Model failed to load, /predict will return 503: %s", e)
    threading.Thread(target=_cleanup_viewer_sessions, daemon=True).start()
    yield


app = FastAPI(title="NeuroScan AI", lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address, default_limits=[])
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Too many requests. Please slow down."})


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response


app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ── Validation helpers ─────────────────────────────────────────────────────
def validate_patient_id(patient_id: str) -> str:
    if not PATIENT_ID_RE.match(patient_id):
        raise HTTPException(status_code=400, detail="Invalid patient ID format.")
    return patient_id


def validate_scan_filename(filename: str) -> str:
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if not filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(status_code=400, detail="Only .nii / .nii.gz files are supported.")
    return filename


def _is_within(base: str, path: str) -> bool:
    base, path = os.path.abspath(base), os.path.abspath(path)
    try:
        return os.path.commonpath([base, path]) == base
    except ValueError:          # different drives on Windows
        return False


async def _read_upload(file: UploadFile) -> tuple:
    if not file.filename or not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(status_code=400, detail="Only .nii or .nii.gz files are supported.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_UPLOAD_MB} MB.")
    suffix = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"
    return content, suffix


# ── Volume loading and slice rendering ─────────────────────────────────────
@lru_cache(maxsize=4)
def _load_canonical(path: str) -> np.ndarray:
    """Load a volume reoriented to RAS so axis 2 is always superior/inferior.
    About two thirds of the cohort is stored as (I,P,L); without this the
    viewer would show those brains rotated. Cached so the slice slider does
    not reload a 70 MB file on every move."""
    img = nib.as_closest_canonical(nib.load(path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError("Volume must be 3D.")
    return data


def _render_axial_png(path: str, slice_idx: int) -> bytes:
    data = _load_canonical(path)
    slice_idx = min(max(slice_idx, 0), data.shape[2] - 1)
    sl = np.rot90(data[:, :, slice_idx])          # anterior up, patient left on image left
    nonzero = sl[sl > 0]
    if nonzero.size:
        vmin, vmax = np.percentile(nonzero, [1, 99])
    else:
        vmin, vmax = float(sl.min()), float(sl.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    img = (np.clip((sl - vmin) / (vmax - vmin), 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _patient_dirs(patient_id: str) -> List[str]:
    dirs = []
    for root in IMAGE_DIRS:
        d = os.path.join(root, patient_id)
        if os.path.isdir(d) and _is_within(root, d):
            dirs.append(d)
    return dirs


def _patient_scans(patient_id: str) -> List[str]:
    files = []
    for d in _patient_dirs(patient_id):
        files += glob.glob(os.path.join(d, "**", "*.nii"), recursive=True)
        files += glob.glob(os.path.join(d, "**", "*.nii.gz"), recursive=True)
    return [f for f in files if "mask" not in f.lower()]


def _find_scan(patient_id: str, scan_filename: str) -> str:
    for f in _patient_scans(patient_id):
        if os.path.basename(f) == scan_filename:
            return f
    raise HTTPException(status_code=404, detail="Scan not found.")


@lru_cache(maxsize=1)
def _dataset_counts() -> dict:
    patients, scans = set(), 0
    for root in IMAGE_DIRS:
        if not os.path.isdir(root):
            continue
        for p in os.listdir(root):
            if os.path.isdir(os.path.join(root, p)):
                patients.add(p)
        scans += len(glob.glob(os.path.join(root, "**", "*.nii"), recursive=True))
    return {"patients": sorted(patients), "scans": scans}


# ── Routes: status ─────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "NeuroScan AI, Alzheimer's MRI analysis API"}


@app.get("/health")
def health_check():
    metrics = _load_metrics()
    return {
        "status": "ok",
        "model_loaded": classifier is not None,
        "model": metrics.get("model") if metrics else None,
        "task": metrics.get("task") if metrics else None,
    }


# ── Routes: prediction ─────────────────────────────────────────────────────
def _classify_locked(path: str) -> dict:
    with _inference_lock:
        return classifier.classify(path)


@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    if classifier is None:
        raise HTTPException(status_code=503, detail="The model is not loaded on the server.")

    content, suffix = await _read_upload(file)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(content)
        tmp.close()
        try:
            shape = nib.load(tmp.name).shape
            if len(shape) < 3:
                raise ValueError
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid NIfTI file. Please upload a 3D T1 MRI scan.")

        # Registration takes 15 to 60 s of CPU; run it off the event loop so the
        # rest of the API stays responsive.
        result = await run_in_threadpool(_classify_locked, tmp.name)
        result["filename"] = file.filename
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again.")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


# ── Routes: dataset and model information ──────────────────────────────────
@app.get("/data/stats")
def get_stats():
    counts = _dataset_counts()
    metrics = _load_metrics() or {}
    cohort = metrics.get("cohort", {})
    return {
        "total_patients": len(counts["patients"]),
        "total_scans": counts["scans"],
        "training_subjects": cohort.get("subjects", 0),
        "model_accuracy": metrics.get("accuracy", 0.0),
        "model_auc": metrics.get("cv_auc_mean", 0.0),
    }


@app.get("/data/model-metrics")
def get_model_metrics():
    metrics = _load_metrics()
    if metrics is None:
        raise HTTPException(status_code=404, detail="Model metrics not found.")
    return metrics


@app.get("/data/demographics")
def get_demographics():
    if not os.path.exists(COHORT_CSV):
        return {"groups": [], "sex": [], "age_bins": []}
    try:
        df = pd.read_csv(COHORT_CSV).drop_duplicates("Subject ID")
        bins = (df["Age"].dropna() // 10 * 10).astype(int)
        return {
            "groups": [{"label": k, "count": int(v)} for k, v in df["Research Group"].value_counts().sort_index().items()],
            "sex": [{"label": k, "count": int(v)} for k, v in df["Sex"].value_counts().sort_index().items()],
            "age_bins": [{"label": f"{b} to {b + 9}", "count": int(v)} for b, v in bins.value_counts().sort_index().items()],
        }
    except Exception as e:
        logger.error("Demographics error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve demographics.")


@app.get("/data/patients")
def get_patients():
    return {"patients": _dataset_counts()["patients"]}


@app.get("/data/image/{patient_id}/scans")
def get_patient_scans(patient_id: str):
    patient_id = validate_patient_id(patient_id)
    return {"scans": sorted({os.path.basename(f) for f in _patient_scans(patient_id)})}


@app.get("/data/image/{patient_id}/{scan_filename}/metadata")
def get_image_metadata(patient_id: str, scan_filename: str):
    path = _find_scan(validate_patient_id(patient_id), validate_scan_filename(scan_filename))
    try:
        shape = _load_canonical(path).shape
    except Exception as e:
        logger.error("Metadata error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Could not read scan metadata.")
    return {"shape": list(shape), "min_slice": 0, "max_slice": shape[2] - 1, "file": scan_filename}


@app.get("/data/image/{patient_id}/{scan_filename}/slice/{slice_idx}")
def get_image_slice(patient_id: str, scan_filename: str, slice_idx: int):
    if slice_idx < 0:
        raise HTTPException(status_code=400, detail="Slice index must be non-negative.")
    path = _find_scan(validate_patient_id(patient_id), validate_scan_filename(scan_filename))
    try:
        return Response(content=_render_axial_png(path, slice_idx), media_type="image/png")
    except Exception as e:
        logger.error("Slice render error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Could not render scan slice.")


# ── Routes: local file viewer ──────────────────────────────────────────────
@app.post("/data/viewer/upload")
@limiter.limit("30/minute")
async def viewer_upload(request: Request, file: UploadFile = File(...)):
    content, suffix = await _read_upload(file)
    session_id = str(uuid.uuid4())
    dest_path = os.path.join(VIEWER_TEMP_DIR, f"{session_id}{suffix}")
    with open(dest_path, "wb") as dest:
        dest.write(content)
    try:
        max_slice = _load_canonical(dest_path).shape[2] - 1
    except Exception:
        os.remove(dest_path)
        raise HTTPException(status_code=400, detail="Invalid or unsupported NIfTI file.")
    _viewer_sessions[session_id] = {"path": dest_path, "created_at": time.time()}
    return {"session_id": session_id, "max_slice": int(max_slice), "filename": file.filename}


@app.get("/data/viewer/{session_id}/slice/{slice_idx}")
def get_viewer_slice(session_id: str, slice_idx: int):
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format.")
    if slice_idx < 0:
        raise HTTPException(status_code=400, detail="Slice index must be non-negative.")
    session = _viewer_sessions.get(session_id)
    if not session or not os.path.exists(session["path"]):
        _viewer_sessions.pop(session_id, None)
        raise HTTPException(status_code=404, detail="Session not found or has expired.")
    try:
        return Response(content=_render_axial_png(session["path"], slice_idx), media_type="image/png")
    except Exception as e:
        logger.error("Viewer slice error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Could not render scan slice.")


# ── Routes: Claude clinical chat ───────────────────────────────────────────
def _build_system_prompt() -> str:
    m = _load_metrics() or {}
    perf = ""
    if m:
        perf = (f"Validated performance (patient-level 5-fold cross-validation on {m.get('cohort', {}).get('subjects', '?')} "
                f"ADNI subjects): ROC AUC {m.get('cv_auc_mean', 0):.2f}, accuracy {m.get('accuracy', 0) * 100:.0f}%, "
                f"AD sensitivity {m['classes'][1]['recall'] * 100:.0f}%, CN specificity {m['classes'][0]['recall'] * 100:.0f}%.")
    return f"""You are NeuroScan AI, a clinical decision support assistant for a research prototype that analyses structural T1 MRI for Alzheimer's disease.

The classifier is binary. It separates cognitively normal (CN) subjects from Alzheimer's disease (AD). It does not detect mild cognitive impairment, and it was never trained on MCI cases, so an MCI patient may be labelled either way.

How the model works: each scan is reoriented, affinely registered to the MNI152 template, skull stripped, and cropped to a box around both hippocampi, the region where Alzheimer's atrophy appears earliest. Three axial slices through that box feed a VGG16 network fine-tuned from ImageNet. {perf}

When a scan result is provided:
1. Restate the classification and the probability in plain clinical language.
2. Explain what it suggests, keeping in mind that roughly one in four predictions is wrong at this performance level.
3. Suggest sensible next steps (cognitive testing, specialist referral, longitudinal imaging).
4. State the limits: research use only, single time point, binary model, trained only on ADNI data.

For general questions, answer concisely about the pipeline, MRI analysis, or Alzheimer's disease. Never present the prediction as a diagnosis. A qualified clinician makes the final decision."""


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=CHAT_MAX_CHARS)


class ScanContext(BaseModel):
    filename: str
    prediction: str
    confidence: float
    class_index: int
    all_probabilities: List[float]


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(min_length=1)
    scan_context: Optional[ScanContext] = None


_chat_client: Optional[anthropic.AsyncAnthropic] = None


def _get_chat_client() -> anthropic.AsyncAnthropic:
    global _chat_client
    if _chat_client is None:
        _chat_client = anthropic.AsyncAnthropic()
    return _chat_client


@app.post("/ai/chat")
@limiter.limit("20/hour")
async def ai_chat(request: Request, req: ChatRequest):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not configured on the server.")

    system = _build_system_prompt()
    if req.scan_context:
        ctx = req.scan_context
        probs = "\n".join(
            f"  {CLASS_NAMES.get(i, f'class {i}')}: {p * 100:.1f}%"
            for i, p in enumerate(ctx.all_probabilities)
        ) or "  not available"
        system += (f"\n\nScan under review: {ctx.filename}\nModel output: {ctx.prediction} "
                   f"({ctx.confidence * 100:.1f}%)\nClass probabilities:\n{probs}")

    history = [{"role": m.role, "content": m.content} for m in req.messages[-CHAT_MAX_MESSAGES:]]
    while history and history[0]["role"] != "user":
        history.pop(0)
    if not history:
        raise HTTPException(status_code=400, detail="The conversation must start with a user message.")

    try:
        response = await _get_chat_client().beta.messages.create(
            model=CHAT_MODEL,
            max_tokens=16000,
            system=system,
            messages=history,
            output_config={"effort": "medium"},
            # Server-side refusal fallback: if a safety classifier declines the
            # request, the API retries it on a fallback model in the same call.
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default"},
        )
    except anthropic.AuthenticationError:
        logger.error("Claude API authentication failed; check ANTHROPIC_API_KEY.")
        raise HTTPException(status_code=500, detail="AI service is misconfigured on the server.")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="The AI service is busy. Please try again in a minute.")
    except anthropic.APIStatusError as e:
        logger.error("Claude API error %s: %s", e.status_code, e.message)
        raise HTTPException(status_code=502, detail="AI service unavailable. Please try again.")
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=503, detail="Could not reach the AI service. Please try again.")

    if response.stop_reason == "refusal":
        return {"content": "I can't help with that request. Please rephrase it as a question about the scan or the model."}

    # Thinking is always on for this model, so the first block may be an empty
    # thinking block. Collect only the text blocks.
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return {"content": text or "I couldn't produce an answer this time. Please try again."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
