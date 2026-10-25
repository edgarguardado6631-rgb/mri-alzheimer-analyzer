# NeuroScan AI: Alzheimer's MRI Analyzer

A research prototype that classifies structural T1 MRI scans as **cognitively normal (CN)** or **Alzheimer's disease (AD)** and explains the result through a Claude-powered clinical chat. Built as the practical component of a master's thesis (MICD, Universidad de Guadalajara).

**Validated performance:** ROC AUC 0.87 ± 0.02 and 80.3% accuracy on 824 ADNI subjects, measured with patient-level 5-fold cross-validation (no subject appears in both training and test).

> Research use only. The model supports, and never replaces, a clinical diagnosis.

## How it works

1. The uploaded scan is reoriented to a standard orientation (RAS).
2. It is affinely registered to the MNI152 brain template, so every brain lines up.
3. The skull is removed with the MNI152 brain mask.
4. A box around both hippocampi is cropped, the region where Alzheimer's atrophy appears first.
5. Three axial slices of that box feed a VGG16 network fine-tuned from ImageNet.

The full research history (leakage discovery, the five architectures compared, and why each preprocessing step exists) is documented in [ml/README.md](ml/README.md).

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 19, TypeScript, Vite, Carbon Design System v11 |
| Backend | Python 3.12, FastAPI, TensorFlow/Keras, dipy, nilearn |
| AI chat | Anthropic Claude API |
| Deployment | Docker, AWS EC2, CloudFront, S3 |

## Project structure

```
project/
├── frontend/            React app (port 3000)
├── ml/                  FastAPI backend, training and preprocessing scripts
│   ├── api.py           REST API
│   ├── inference.py     Deployed prediction pipeline
│   ├── models/          final_full_roi_m{1,2,3}.keras + final_metrics.json (models not in git)
│   └── results/         metrics, figures and logs from every experiment
├── data/                ADNI data and cohort files (not in git, see below)
└── docker-compose.yml
```

## Getting started

### Docker

```bash
docker compose up --build
```

Frontend at http://localhost:3000, API docs at http://localhost:8000/docs.

> Not yet tested with the three-model ensemble. The backend image built from `ml/Dockerfile` is the one deployed and verified on AWS; the Compose setup (which also builds the frontend and mounts `data/`) has not been run since the switch.

### AWS

`deploy.py` starts the EC2 backend, points CloudFront at its current address, uploads the three models, rebuilds the container and publishes the frontend to S3:

```powershell
venv\Scripts\python.exe deploy.py up       # deploy everything
venv\Scripts\python.exe deploy.py status   # instance state and /health
venv\Scripts\python.exe deploy.py down     # stop the instance when not in use
```

It needs the AWS CLI configured, `boto3`, and `neuroscan-key.pem` in the project root (never committed). `ec2-setup.sh` is the one-time setup of a new instance.

### Local (Windows)

```powershell
cd ml
..\venv\Scripts\python.exe -m uvicorn api:app --port 8000
```

```powershell
cd frontend
npm install
npm run dev
```

The backend needs about 15 seconds to load TensorFlow, the model and the MNI template. A prediction takes 15 to 60 seconds on CPU because each scan is registered to the template.

## Configuration

Copy `ml/.env.example` to `ml/.env` and set `ANTHROPIC_API_KEY`. The other variables are optional.

## Model and data

Neither the model weights nor the MRI data are in the repository.

* **Data:** ADNI data can only be obtained from https://adni.loni.usc.edu under its Data Use Agreement, which does not allow redistribution. Place the downloaded trees under `data/` as described in [ml/README.md](ml/README.md).
* **Model:** copy `final_full_roi_m1.keras`, `final_full_roi_m2.keras`, `final_full_roi_m3.keras` and `final_metrics.json` into `ml/models/`, or rebuild them with the steps in [ml/README.md](ml/README.md).

Without a model, `/predict` returns HTTP 503; every other endpoint still works. A prediction runs as a background job because registration can take close to CloudFront's 60 s origin timeout; the frontend polls `/predict/{job_id}` every 2 s.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Upload a `.nii` or `.nii.gz` T1 scan; starts the analysis and returns `job_id` (HTTP 202) |
| `/predict/{job_id}` | GET | Job status (`queued`, `running`, `done`, `error`) and, when done, the CN/AD probabilities |
| `/ai/chat` | POST | Conversation plus optional scan context; returns Claude's interpretation |
| `/data/stats` | GET | Archive size, training cohort size, AUC and accuracy |
| `/data/demographics` | GET | Diagnosis, sex and age distribution of the training cohort |
| `/data/model-metrics` | GET | Full validation report of the deployed model |
| `/data/patients` | GET | Patients available in the viewer |
| `/health` | GET | Server and model status |

## License

MIT for the code. ADNI data is subject to the ADNI Data Use Agreement.
