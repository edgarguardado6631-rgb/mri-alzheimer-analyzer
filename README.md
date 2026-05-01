# MRI Alzheimer Analyzer

A web application for Alzheimer's disease classification from MRI scans using a fine-tuned VGG16 CNN. Upload a `.nii` or `.nii.gz` MRI file, get an instant 3-class prediction (CN / MCI / AD), and receive a Claude-powered clinical interpretation directly in the chat.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, TypeScript, Vite, Carbon Design System v11 |
| Backend | Python, FastAPI, TensorFlow/Keras, Anthropic Claude API |
| Containerization | Docker, Docker Compose |

## Project Structure

```
mri-alzheimer-analyzer/
├── frontend/          # React + TypeScript + Vite app (port 3000)
├── ml/                # FastAPI backend + CNN model (port 8000)
│   ├── api.py
│   ├── train_cnn.py
│   └── .env          # API keys (not committed — see below)
├── docker-compose.yml
└── README.md
```

## Getting Started

### Option 1 — Docker (recommended)

```bash
docker-compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/docs

### Option 2 — Local (Windows PowerShell)

**1. Backend**

```powershell
cd E:\code\project\ml
..\venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Wait ~15 seconds for TensorFlow to load the model before making requests.

**2. Frontend** (new terminal)

```powershell
cd E:\code\project\frontend
npm run dev
```

Open http://localhost:3000

### Option 3 — Local (macOS / Linux)

**Backend**

```bash
cd ml
source ../venv/bin/activate
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

## Environment Variables

Create `ml/.env` (copy from `ml/.env.example`):

```env
# Required for Claude-powered clinical interpretation
ANTHROPIC_API_KEY=sk-ant-...

# Production only — restrict CORS origins
# ALLOWED_ORIGINS=https://your-domain.com
```

Get your API key at https://console.anthropic.com

## Model Weights

The trained model (`alzheimer_cnn_model.h5`, ~122 MB) is **not included** in the repository. To use the app:

- **Option A — Train it yourself:**
  ```powershell
  cd E:\code\project\ml
  ..\venv\Scripts\python.exe train_cnn.py
  ```
  Training takes ~10–20 minutes on CPU (400 samples, 15 epochs).

- **Option B — Download pre-trained weights** and place `alzheimer_cnn_model.h5` inside `ml/`.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Upload `.nii`/`.nii.gz` — returns CNN classification + confidence |
| `/ai/chat` | POST | Send conversation + scan context — returns Claude clinical interpretation |
| `/data/stats` | GET | Cohort summary (patients, scans, accuracy) |
| `/data/demographics` | GET | Class / sex / age distribution |
| `/data/model-metrics` | GET | Per-class precision, recall, F1 from last training run |
| `/health` | GET | Server + model load status |
| `/docs` | GET | Interactive Swagger API documentation |

## License

MIT
