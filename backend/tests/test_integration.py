"""
Integration tests covering the full pipeline:

    Upload → OCR → Feature Engineering → ML Prediction → Advice Response

These tests train a small model in a temp directory and run the full
pipeline without mocking individual modules (except EasyOCR which
requires a GPU / heavy model download).
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_model_dir(tmp_path_factory):
    """
    Generate a small dataset, train a model and return the paths dict.
    Runs once per module to avoid repetitive training overhead.
    """
    tmp = tmp_path_factory.mktemp("integration")
    csv = tmp / "data.csv"
    model = tmp / "addiction_model.pkl"
    metrics = tmp / "metrics.json"

    # Patch settings before importing trainer to avoid polluting real dirs
    import app.config as cfg

    cfg.settings.dataset_path = str(csv)
    cfg.settings.model_path = str(model)
    cfg.settings.metrics_path = str(metrics)
    cfg.settings.test_size = 0.2
    cfg.settings.cv_folds = 3

    import app.ml.trainer as trainer_mod
    import app.ml.predictor as predictor_mod
    import app.ml.dataset_generator as gen_mod

    trainer_mod.settings.dataset_path = str(csv)
    trainer_mod.settings.model_path = str(model)
    trainer_mod.settings.metrics_path = str(metrics)
    trainer_mod.settings.test_size = 0.2
    trainer_mod.settings.cv_folds = 3

    predictor_mod.settings.model_path = str(model)
    predictor_mod.settings.metrics_path = str(metrics)
    gen_mod.settings.dataset_path = str(csv)

    # Generate & train
    from app.ml.dataset_generator import generate_dataset, save_dataset

    save_dataset(generate_dataset(n_samples=400))

    from app.ml.trainer import train

    train()

    # Reset cached model so it reloads from new path
    predictor_mod._model = None

    return {"csv": csv, "model": model, "metrics": metrics, "tmp": tmp}


@pytest.fixture(scope="module")
def client(trained_model_dir):
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _make_png_bytes() -> bytes:
    img = Image.new("RGB", (400, 200), color=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# Fake OCR result to avoid needing actual EasyOCR model in CI
_FAKE_OCR = {
    "screen_time_hours": 7.0,
    "social_media_hours": 3.5,
    "gaming_hours": 1.0,
    "unlock_count": 110,
    "night_usage": 1,
    "raw_text": "Screen Time 7h Instagram 3h 30m PUBG 1h unlocked 110 times 23:00",
    "confidence": 0.87,
}


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline tests
# ──────────────────────────────────────────────────────────────────────────────

class TestFullPipeline:
    def test_health_after_model_load(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_metrics_available_after_training(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert "RandomForest" in data["models"]

    def test_model_accuracy_above_threshold(self, client):
        r = client.get("/metrics")
        rf_acc = r.json()["models"]["RandomForest"]["accuracy"]
        assert rf_acc >= 0.75, f"Accuracy {rf_acc:.4f} is below 75 % threshold"

    def test_upload_returns_structured_features(self, client):
        with patch("app.api.routes.extract_from_bytes", return_value=_FAKE_OCR):
            r = client.post(
                "/upload",
                files={"file": ("screen.png", _make_png_bytes(), "image/png")},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["screen_time_hours"] == 7.0
        assert data["unlock_count"] == 110
        assert data["night_usage"] == 1

    def test_predict_end_to_end(self, client):
        """Upload → OCR (mocked) → real ML model → rule-based advice."""
        with patch("app.api.routes.extract_from_bytes", return_value=_FAKE_OCR):
            with patch("app.api.routes.get_advice", new=AsyncMock(
                return_value={"advice": "Test advice", "source": "rule-based"}
            )):
                r = client.post(
                    "/predict",
                    files={"file": ("screen.png", _make_png_bytes(), "image/png")},
                )
        assert r.status_code == 200
        data = r.json()
        assert data["label"] in {"Low", "Moderate", "High"}
        assert 0 <= data["addiction_score"]
        proba_sum = sum(data["probabilities"].values())
        assert abs(proba_sum - 1.0) < 1e-3
        assert "advice" in data

    def test_predict_manual_end_to_end(self, client):
        body = {
            "screen_time_hours": 9.0,
            "social_media_hours": 5.0,
            "gaming_hours": 2.0,
            "unlock_count": 200,
            "night_usage": 1,
        }
        with patch("app.api.routes.get_advice", new=AsyncMock(
            return_value={"advice": "High-risk advice", "source": "rule-based"}
        )):
            r = client.post("/predict/manual", json=body)
        assert r.status_code == 200
        data = r.json()
        # High usage should predict High or Moderate
        assert data["label"] in {"High", "Moderate"}

    def test_predict_low_usage_end_to_end(self, client):
        low_ocr = {**_FAKE_OCR,
                   "screen_time_hours": 1.5,
                   "social_media_hours": 0.5,
                   "unlock_count": 20,
                   "night_usage": 0}
        with patch("app.api.routes.extract_from_bytes", return_value=low_ocr):
            with patch("app.api.routes.get_advice", new=AsyncMock(
                return_value={"advice": "Good job!", "source": "rule-based"}
            )):
                r = client.post(
                    "/predict",
                    files={"file": ("screen.png", _make_png_bytes(), "image/png")},
                )
        assert r.status_code == 200
        assert r.json()["label"] == "Low"

    def test_feature_consistency_between_ocr_and_prediction(self, client):
        """Features returned by /predict should match those sent to the model."""
        with patch("app.api.routes.extract_from_bytes", return_value=_FAKE_OCR):
            with patch("app.api.routes.get_advice", new=AsyncMock(
                return_value={"advice": "ok", "source": "rule-based"}
            )):
                r = client.post(
                    "/predict",
                    files={"file": ("screen.png", _make_png_bytes(), "image/png")},
                )
        data = r.json()
        assert data["features"]["screen_time_hours"] == _FAKE_OCR["screen_time_hours"]
        assert data["features"]["unlock_count"] == _FAKE_OCR["unlock_count"]
