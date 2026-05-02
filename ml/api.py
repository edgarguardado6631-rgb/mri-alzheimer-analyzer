from dotenv import load_dotenv
import pathlib
load_dotenv(pathlib.Path(__file__).parent / '.env', override=True)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
import numpy as np
import nibabel as nib
import os
import shutil
import glob
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = FastAPI()

# ALLOWED_ORIGINS: comma-separated list of allowed origins.
# Defaults to "*" for local dev. Set to your CloudFront URL in production.
# e.g. ALLOWED_ORIGINS=https://d1234abcd.cloudfront.net
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in _raw_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "alzheimer_cnn_model.h5")
model = None

class _CompatDense(tf.keras.layers.Dense):
    """Keras version compatibility shim — strips unknown 'quantization_config'
    argument that newer Keras versions add to Dense layer configs."""
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
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
    else:
        print("Model file not found. Predictions will be mocked until training is run.")

@app.get("/")
def read_root():
    return {"message": "Alzheimer's MRI Analysis API"}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".nii") and not file.filename.endswith(".nii.gz"):
         raise HTTPException(status_code=400, detail="Only .nii or .nii.gz files are supported.")
    
    # Save temp
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 3-class label map — must match LABEL_MAP used during training
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

    try:
        # Load and preprocess — matches training pipeline in train_cnn.py
        img = nib.load(temp_path)
        data = img.get_fdata()

        # Axis-aware slice selection (matches notebook / train_cnn.py)
        axcodes = nib.aff2axcodes(img.affine)
        si_axis = next((idx for idx, c in enumerate(axcodes) if c in ['S', 'I']), 2)
        mid_slice = data.shape[si_axis] // 2
        if si_axis == 0:
            slice_img = data[mid_slice, :, :]
        elif si_axis == 1:
            slice_img = data[:, mid_slice, :]
        else:
            slice_img = data[:, :, mid_slice]

        # Resize then normalize (add .numpy() to match training pipeline)
        slice_img = tf.image.resize(slice_img[..., np.newaxis], (128, 128)).numpy()
        vmin, vmax = slice_img.min(), slice_img.max()
        slice_img = (slice_img - vmin) / (vmax - vmin + 1e-8)
        input_data = np.array([slice_img])

        # Predict
        if model:
            prediction = model.predict(input_data)
            class_idx = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][class_idx])
            label = LABEL_MAP.get(class_idx, f"Unknown class {class_idx}")
            analysis = ANALYSIS_MAP.get(class_idx, "Further clinical correlation recommended.")
        else:
            import random
            class_idx = random.randint(0, 2)
            confidence = random.uniform(0.7, 0.99)
            label = f"MOCK — {LABEL_MAP[class_idx]}"
            analysis = "This is a mock result. Train the model to get real predictions."

        return {
            "prediction": label,
            "confidence": confidence,
            "analysis": analysis,
            "class_index": class_idx,
            "all_probabilities": prediction[0].tolist() if model else [],
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/data/stats")
def get_stats():
    base_path = "data"
    if not os.path.exists(base_path) and os.path.exists("../data"):
        base_path = "../data"
        
    adni_path = os.path.join(base_path, "Test 2", "ADNI")
    
    total_patients = 0
    total_scans = 0
    
    if os.path.exists(adni_path):
        # The structure is data/Test 2/ADNI/<patient_id>/...
        # Count patients
        patients_dirs = [d for d in os.listdir(adni_path) if os.path.isdir(os.path.join(adni_path, d))]
        total_patients = len(patients_dirs)
        
        # Count total scans (NIfTI files)
        nii_files = glob.glob(os.path.join(adni_path, "**", "*.nii"), recursive=True)
        nii_gz_files = glob.glob(os.path.join(adni_path, "**", "*.nii.gz"), recursive=True)
        total_scans = len(nii_files) + len(nii_gz_files)
        
    return {
        "total_patients": total_patients,
        "scans_processed": total_scans,
        "model_accuracy": 0.82 # VGG16 fine-tuned, 15 epochs, 80-sample test set
    }

@app.get("/data/model-metrics")
def get_model_metrics():
    import json
    metrics_path = os.path.join(BASE_DIR, "model_metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Model metrics not found. Run training first.")
    with open(metrics_path, "r") as f:
        return json.load(f)


@app.get("/data/demographics")
def get_demographics():
    base_path = "data"
    if not os.path.exists(base_path) and os.path.exists("../data"):
        base_path = "../data"

    csv_files = glob.glob(os.path.join(base_path, "*.csv"))
    if not csv_files:
        return {"groups": [], "sex": [], "age_bins": []}

    import csv
    from collections import defaultdict

    group_counts: dict = defaultdict(int)
    sex_counts: dict = defaultdict(int)
    age_values: list = []
    seen_subjects: set = set()

    with open(csv_files[0], newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject = row.get("Subject", "").strip()
            if subject in seen_subjects:
                continue
            seen_subjects.add(subject)
            group = row.get("Group", "").strip()
            sex = row.get("Sex", "").strip()
            age_str = row.get("Age", "").strip()
            if group:
                group_counts[group] += 1
            if sex:
                sex_counts[sex] += 1
            if age_str.isdigit():
                age_values.append(int(age_str))

    # Build age distribution in 10-year bins
    age_bins: dict = defaultdict(int)
    for age in age_values:
        bin_start = (age // 10) * 10
        label = f"{bin_start}–{bin_start + 9}"
        age_bins[label] += 1

    return {
        "groups": [{"label": k, "count": v} for k, v in sorted(group_counts.items())],
        "sex": [{"label": k, "count": v} for k, v in sorted(sex_counts.items())],
        "age_bins": [{"label": k, "count": v} for k, v in sorted(age_bins.items())],
    }


@app.get("/data/patients")
def get_patients():
    base_path = "data"
    if not os.path.exists(base_path) and os.path.exists("../data"):
        base_path = "../data"
    
    # The actual data seems to be in data/Test 2/ADNI/
    adni_path = os.path.join(base_path, "Test 2", "ADNI")
    if os.path.exists(adni_path):
        patients = [d for d in os.listdir(adni_path) if os.path.isdir(os.path.join(adni_path, d))]
        return {"patients": sorted(patients)}
    
    # Fallback to current behavior if structure is different
    files = glob.glob(os.path.join(base_path, "**", "*.nii"), recursive=True) + \
            glob.glob(os.path.join(base_path, "**", "*.nii.gz"), recursive=True)
    patients = set()
    for f in files:
        # Try to extract subject ID (e.g., 002_S_0413)
        parts = f.split(os.sep)
        for part in parts:
            if "_" in part and len(part.split("_")) == 3:
                patients.add(part)
    return {"patients": sorted(list(patients))}

@app.get("/data/image/{patient_id}/scans")
def get_patient_scans(patient_id: str):
    base_path = "data"
    if not os.path.exists(base_path) and os.path.exists("../data"):
        base_path = "../data"
        
    adni_path = os.path.join(base_path, "Test 2", "ADNI", patient_id)
    if os.path.exists(adni_path):
        pattern = os.path.join(adni_path, "**", "*.nii")
        files = glob.glob(pattern, recursive=True) + glob.glob(os.path.join(adni_path, "**", "*.nii.gz"), recursive=True)
    else:
        pattern = os.path.join(base_path, "**", f"*{patient_id}*.nii")
        files = glob.glob(pattern, recursive=True) + glob.glob(os.path.join(base_path, "**", f"*{patient_id}*.nii.gz"), recursive=True)

    scans = [os.path.basename(f) for f in files]
    return {"scans": sorted(list(set(scans)))}

@app.get("/data/image/{patient_id}/{scan_filename}/metadata")
def get_image_metadata(patient_id: str, scan_filename: str):
    base_path = "data"
    if not os.path.exists(base_path) and os.path.exists("../data"):
        base_path = "../data"
        
    # Search recursively for the specific file
    adni_path = os.path.join(base_path, "Test 2", "ADNI", patient_id)
    if os.path.exists(adni_path):
        pattern = os.path.join(adni_path, "**", scan_filename)
        files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(base_path, "**", scan_filename)
        files = glob.glob(pattern, recursive=True)
        
    if not files:
        raise HTTPException(status_code=404, detail=f"File {scan_filename} not found for patient {patient_id}")
        
    # Pick the first one for now
    file_path = files[0]
    try:
        img = nib.load(file_path)
        return {
            "shape": img.shape,
            "min_slice": 0,
            "max_slice": img.shape[2] - 1,
            "file": os.path.basename(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/image/{patient_id}/{scan_filename}/slice/{slice_idx}")
def get_image_slice(patient_id: str, scan_filename: str, slice_idx: int):
    base_path = "data"
    if not os.path.exists(base_path) and os.path.exists("../data"):
        base_path = "../data"
        
    # Search recursively for the specific file
    adni_path = os.path.join(base_path, "Test 2", "ADNI", patient_id)
    if os.path.exists(adni_path):
        pattern = os.path.join(adni_path, "**", scan_filename)
        files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(base_path, "**", scan_filename)
        files = glob.glob(pattern, recursive=True)
        
    if not files:
        raise HTTPException(status_code=404, detail=f"File {scan_filename} not found for patient {patient_id}")
        
    file_path = files[0]
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        
        if slice_idx < 0 or slice_idx >= data.shape[2]:
             slice_idx = data.shape[2] // 2 # Fallback to middle if out of bounds
             
        slice_img = data[:, :, slice_idx]
        
        # Normalize for display
        slice_img = np.rot90(slice_img)
        plt.figure(figsize=(4, 4))
        plt.imshow(slice_img, cmap='gray')
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        print(f"Error serving slice: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
async def ai_chat(request: ChatRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not set on the server.")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        system = CLINICAL_SYSTEM_PROMPT
        if request.scan_context:
            ctx = request.scan_context
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
            messages=[{"role": m.role, "content": m.content} for m in request.messages],
        )
        return {"content": response.content[0].text}

    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
