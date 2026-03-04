# 📱 Smartphone Addiction Predictor

A production-ready, fully local ML system that analyses smartphone
screen-time screenshots, extracts behavioural features via OCR, and
predicts your addiction risk level (Low / Moderate / High) – with
optional personalised advice from a local LLM.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend                           │
│  Upload → Loading Animation → Dashboard (Radar Chart + Advice) │
└─────────────────────┬───────────────────────────────────────────┘
                      │  HTTP (multipart/form-data)
┌─────────────────────▼───────────────────────────────────────────┐
│                     FastAPI Backend                             │
│                                                                 │
│  POST /upload     POST /predict     GET /health  GET /metrics   │
│        │                │                                       │
│        ▼                ▼                                       │
│   ┌─────────┐    ┌─────────────┐                               │
│   │ EasyOCR │    │ EasyOCR     │                               │
│   │ extractor│   │ extractor   │                               │
│   └─────────┘    └──────┬──────┘                               │
│                         │ structured features                   │
│                   ┌──────▼──────┐                               │
│                   │ Random Forest│                              │
│                   │  classifier  │                              │
│                   └──────┬──────┘                               │
│                          │ label + probabilities                │
│                   ┌──────▼──────┐                               │
│                   │ LLM Advisor │ (Ollama → rule-based fallback)│
│                   └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
              (optional)
┌─────────────────────────────────────────────────────────────────┐
│                   Ollama (local LLM)                            │
│                   llama3 / mistral / phi3                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
zak/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app factory & lifespan
│   │   ├── config.py            # Pydantic settings (reads .env)
│   │   ├── ocr/
│   │   │   └── extractor.py     # EasyOCR + regex parser
│   │   ├── ml/
│   │   │   ├── dataset_generator.py  # Synthetic data factory
│   │   │   ├── features.py      # Score formula & feature builder
│   │   │   ├── trainer.py       # LR + RF training pipeline
│   │   │   └── predictor.py     # Model loading & inference
│   │   ├── api/
│   │   │   └── routes.py        # All FastAPI endpoints
│   │   └── llm/
│   │       └── advisor.py       # Ollama client + rule-based fallback
│   ├── tests/
│   │   ├── test_ocr.py          # Unit tests – OCR
│   │   ├── test_ml.py           # Unit tests – ML
│   │   ├── test_api.py          # API tests (mocked)
│   │   └── test_integration.py  # Full pipeline tests
│   ├── data/                    # Generated CSV (git-ignored)
│   ├── models/                  # Saved .pkl + metrics.json
│   ├── requirements.txt
│   ├── pyproject.toml           # pytest configuration
│   └── conftest.py
├── frontend/
│   ├── src/
│   │   ├── main.jsx
│   │   ├── App.jsx
│   │   ├── App.module.css
│   │   ├── index.css
│   │   └── components/
│   │       ├── UploadSection.jsx   # Drag-and-drop upload
│   │       ├── Dashboard.jsx       # Results dashboard
│   │       ├── BehaviorRadar.jsx   # Recharts radar chart
│   │       └── Advice.jsx          # Personalised advice panel
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── nginx.conf
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Dataset

Synthetic dataset with **1 200 samples** (configurable).

| Feature | Range | Notes |
|---|---|---|
| `screen_time_hours` | 0.5 – 16 h | 3-archetype mixture model |
| `social_media_hours` | 0 – 12 h | Correlated with screen time |
| `gaming_hours` | 0 – 8 h | Right-skewed distribution |
| `unlock_count` | 5 – 400 | Scales with screen time |
| `night_usage` | 0 / 1 | Probability tied to social media |

### Addiction Score Formula

```
score = 0.4 × screen_time_hours
      + 0.3 × social_media_hours
      + 0.2 × (unlock_count / 100)
      + 0.1 × night_usage
```

| Score | Label |
|---|---|
| < 3.0 | Low |
| 3.0 – 4.99 | Moderate |
| ≥ 5.0 | High |

---

## ML Training

Two classifiers are trained and cross-validated:

| Model | CV Strategy | Notes |
|---|---|---|
| Logistic Regression | 5-fold stratified | baseline |
| **Random Forest** | 5-fold stratified | **primary model** |

Saved artefacts:
- `backend/models/addiction_model.pkl` – joblib-serialised Random Forest
- `backend/models/metrics.json` – accuracy, F1, confusion matrix, feature importances

---

## Quick Start (Local)

### Prerequisites

- Python 3.10+
- Node.js 18+
- (Optional) [Ollama](https://ollama.com) for AI-generated advice

### 1. Backend setup

```bash
cd backend

# Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# Copy environment config
cp ../.env.example .env

# Generate dataset & train model
python -m app.ml.dataset_generator
python -m app.ml.trainer

# Start API server
uvicorn app.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### 2. Frontend setup

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:3000
```

### 3. Optional – Ollama LLM

```bash
# Install Ollama from https://ollama.com
ollama pull llama3
ollama serve   # starts on http://localhost:11434
```

The backend auto-detects Ollama and falls back to rule-based advice if unavailable.

---

## Docker Deployment

```bash
# Backend + Frontend only
docker compose up --build

# Include Ollama
docker compose --profile llm up --build
```

Services:
- Frontend → http://localhost:80
- Backend API → http://localhost:8000
- Ollama → http://localhost:11434 (profile: llm)

---

## Running Tests

```bash
cd backend

# All tests with coverage
pytest

# Specific suite
pytest tests/test_ocr.py -v
pytest tests/test_ml.py -v
pytest tests/test_api.py -v
pytest tests/test_integration.py -v

# Coverage report only
pytest --cov=app --cov-report=html
# Open htmlcov/index.html
```

Minimum coverage threshold: **80 %** (enforced via `pyproject.toml`).

---

## API Reference

### `GET /health`
```json
{ "status": "ok", "version": "1.0.0" }
```

### `GET /metrics`
Returns full training metrics JSON.

### `POST /upload`
- Body: `multipart/form-data` with field `file` (image)
- Returns OCR-extracted features:
```json
{
  "screen_time_hours": 7.5,
  "social_media_hours": 3.8,
  "gaming_hours": 0.75,
  "unlock_count": 112,
  "night_usage": 1,
  "raw_text": "...",
  "confidence": 0.88
}
```

### `POST /predict`
- Body: same multipart image upload
- Returns full prediction:
```json
{
  "label": "High",
  "probabilities": { "Low": 0.04, "Moderate": 0.18, "High": 0.78 },
  "addiction_score": 5.83,
  "features": { "screen_time_hours": 7.5, "..." : "..." },
  "advice": "• Set strict app limits...",
  "advice_source": "rule-based"
}
```

### `POST /predict/manual`
- Body: JSON with explicit feature values (no image required)
```json
{
  "screen_time_hours": 8.0,
  "social_media_hours": 4.0,
  "gaming_hours": 1.5,
  "unlock_count": 180,
  "night_usage": 1
}
```

---

## Testing Details

| Suite | Coverage |
|---|---|
| `test_ocr.py` | `_text_to_hours`, `_detect_night_usage`, `_parse_text`, `extract_from_bytes` (mocked EasyOCR) |
| `test_ml.py` | Score formula, label thresholds, feature vector, dataset generation, model training, accuracy ≥ 75 % |
| `test_api.py` | All endpoints, validation errors, missing file, corrupted image, 503 scenarios |
| `test_integration.py` | Full pipeline with real trained model (EasyOCR mocked), feature consistency, accuracy threshold |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DEBUG` | `false` | Enable debug logging |
| `MODEL_PATH` | `models/addiction_model.pkl` | Path to saved model |
| `METRICS_PATH` | `models/metrics.json` | Path to metrics file |
| `DATASET_PATH` | `data/smartphone_addiction.csv` | Training data path |
| `TEST_SIZE` | `0.2` | Train/test split ratio |
| `CV_FOLDS` | `5` | Cross-validation folds |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base |
| `OLLAMA_MODEL` | `llama3` | Model name for Ollama |
| `OLLAMA_TIMEOUT` | `30` | Ollama request timeout (s) |
| `MAX_UPLOAD_SIZE_MB` | `10` | Max image upload size |

---

## License

MIT
