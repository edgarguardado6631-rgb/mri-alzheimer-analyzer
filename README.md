# MRI Alzheimer Analyzer

A web application for Alzheimer's disease classification from MRI scans using a Convolutional Neural Network (CNN). Upload a `.nii` or `.nii.gz` MRI file and get an instant AI-powered prediction.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, TypeScript, Vite, Carbon Design System |
| Backend | Python, FastAPI, TensorFlow/Keras |
| Containerization | Docker, Docker Compose |

## Project Structure

```
mri-alzheimer-analyzer/
├── frontend/        # React + TypeScript + Vite app (port 3000)
├── ml/              # FastAPI backend + CNN model (port 8000)
├── docker-compose.yml
└── requirements.txt
```

## Getting Started

### Option 1 — Docker (recommended)

```bash
docker-compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

### Option 2 — Local

**Backend**

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
cd ml
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

## Model Weights

The trained model file (`alzheimer_cnn_model.h5`) is **not included** in this repository due to its size (~122 MB). To use the application:

- **Option A:** Train the model yourself using the provided training script:
  ```bash
  cd ml
  python train_cnn.py
  ```
- **Option B:** Download the pre-trained weights and place the `.h5` file inside the `ml/` directory.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Upload a `.nii`/`.nii.gz` MRI file and receive a classification result |
| `/docs` | GET | Interactive Swagger API documentation |

## License

MIT
