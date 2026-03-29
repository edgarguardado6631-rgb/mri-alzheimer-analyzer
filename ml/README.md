# AI MRI Analysis - Backend

This directory contains the Python backend for the AI MRI Analysis application. It serves a REST API built with FastAPI that provides data and machine learning predictions to the frontend.

## Prerequisites

- [Python 3.10+](https://www.python.org/downloads/)
- [Docker & Docker Compose](https://www.docker.com/products/docker-desktop/) (Optional, for containerized execution)

## Setup and Installation

### 1. Create a Virtual Environment (Recommended)

Navigate to the root of the project (`e:\code\project`) and create a Python virtual environment:

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

**On Windows:**
```bash
.\venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

With the virtual environment active, install the required packages:

```bash
pip install -r requirements.txt
```

## Running the API Locally

Make sure you are in the `ml` directory:

```bash
cd ml
```

Run the server using `uvicorn` or by directly executing the python file. 

If you have already activated your virtual environment:

```bash
# Option 1 (using direct python execution)
python api.py

# Option 2 (using uvicorn directly for development)
# This will enable auto-reload on code changes
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Alternative: Running directly through the Virtual Environment (without activating it first)**
If you haven't activated your environment, you can run the files directly by referencing the Python executable inside your `venv` folder (run this from the project root `e:\code\project`):

**On Windows (PowerShell/CMD):**
```bash
.\venv\Scripts\python ml\api.py
```

**On macOS / Linux / WSL (Bash):**
```bash
# If the venv was created in Linux/WSL:
./venv/bin/python ml/api.py

# If the venv was created in Windows but you are using Bash:
./venv/Scripts/python ml/api.py
```

The API will be available at `http://localhost:8000`. 
You can view the interactive API documentation at: `http://localhost:8000/docs`.

### Stopping the API
Press `Ctrl + C` in the terminal where the API is running to stop it.

## Training the CNN Model

If you have updated the dataset or modified the Convolutional Neural Network architecture and need to generate a new `alzheimer_cnn_model.h5` model file, you can run the training script.

If your virtual environment is **active** (run from the `ml` directory):
```bash
python train_cnn.py
```

If your virtual environment is **NOT active** (run from the `e:\code\project` root directory):

**On Windows (PowerShell/CMD):**
```bash
.\venv\Scripts\python ml\train_cnn.py
```

**On macOS / Linux / WSL (Bash):**
```bash
# If the venv was created in Linux/WSL:
./venv/bin/python ml/train_cnn.py

# If the venv was created in Windows but you are using Bash:
./venv/Scripts/python ml/train_cnn.py
```

## Running with Docker

If you prefer to run the entire stack (including the frontend) using Docker, you can use the `docker-compose.yml` file located in the root directory.

From the root directory (`e:\code\project`), run:

```bash
docker-compose up --build
```

This will build both backend and frontend images and start the containers. The backend will map to port `8000` and the frontend to port `3000`.

## Project Structure

- `api.py`: The main FastAPI application entry point, containing API routes, image loaders, and the CNN prediction logic.
- `train_cnn.py`: A script to train the CNN model.
- `Dockerfile`: Defines the Docker image setup for the backend.
- `alzheimer_cnn_model.h5`: The trained TensorFlow/Keras model file (if it exists).
