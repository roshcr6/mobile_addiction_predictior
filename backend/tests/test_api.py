"""
API tests for all FastAPI endpoints.

Uses HTTPX TestClient with mocked OCR and ML predictor so tests run
without a trained model or GPU.
"""
from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """HTTPX sync test client; startup/shutdown events are skipped."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _make_png_bytes(label: str = "Screen Time 5h 30m\nInstagram 2h") -> bytes:
    img = Image.new("RGB", (400, 120), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_MOCK_OCR_RESULT = {
    "screen_time_hours": 5.5,
    "social_media_hours": 2.0,
    "gaming_hours": 0.5,
    "unlock_count": 80,
    "night_usage": 0,
    "raw_text": "Screen Time 5h 30m",
    "confidence": 0.91,
}

_MOCK_PREDICTION = {
    "label": "Moderate",
    "probabilities": {"High": 0.15, "Low": 0.20, "Moderate": 0.65},
    "addiction_score": 3.26,
    "features": {
        "screen_time_hours": 5.5,
        "social_media_hours": 2.0,
        "gaming_hours": 0.5,
        "unlock_count": 80.0,
        "night_usage": 0.0,
    },
}

_MOCK_ADVICE = {"advice": "Take more breaks.", "source": "rule-based"}


# ──────────────────────────────────────────────────────────────────────────────
# /health
# ──────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_has_status_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_has_version(self, client):
        r = client.get("/health")
        assert "version" in r.json()


# ──────────────────────────────────────────────────────────────────────────────
# /metrics
# ──────────────────────────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_returns_503_when_no_model(self, client):
        with patch("app.api.routes.load_metrics", return_value={"error": "No model"}):
            r = client.get("/metrics")
        assert r.status_code == 503

    def test_returns_200_with_metrics(self, client):
        fake_metrics = {"best_model": "RandomForest", "models": {}}
        with patch("app.api.routes.load_metrics", return_value=fake_metrics):
            r = client.get("/metrics")
        assert r.status_code == 200
        assert r.json()["best_model"] == "RandomForest"


# ──────────────────────────────────────────────────────────────────────────────
# POST /upload
# ──────────────────────────────────────────────────────────────────────────────

class TestUploadEndpoint:
    def test_valid_image_returns_200(self, client):
        with patch("app.api.routes.extract_from_bytes", return_value=_MOCK_OCR_RESULT):
            r = client.post(
                "/upload",
                files={"file": ("screen.png", _make_png_bytes(), "image/png")},
            )
        assert r.status_code == 200

    def test_response_has_screen_time(self, client):
        with patch("app.api.routes.extract_from_bytes", return_value=_MOCK_OCR_RESULT):
            r = client.post(
                "/upload",
                files={"file": ("screen.png", _make_png_bytes(), "image/png")},
            )
        assert r.json()["screen_time_hours"] == 5.5

    def test_non_image_returns_422(self, client):
        r = client.post(
            "/upload",
            files={"file": ("data.txt", b"hello", "text/plain")},
        )
        assert r.status_code == 422

    def test_empty_file_returns_422(self, client):
        r = client.post(
            "/upload",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert r.status_code == 422

    def test_corrupted_image_returns_422(self, client):
        with patch(
            "app.api.routes.extract_from_bytes",
            side_effect=ValueError("Cannot open image"),
        ):
            r = client.post(
                "/upload",
                files={"file": ("bad.png", b"\x00\x01\x02", "image/png")},
            )
        assert r.status_code == 422

    def test_missing_file_returns_422(self, client):
        r = client.post("/upload")
        assert r.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# POST /predict
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def _post(self, client, extra_patches=None):
        patches = [
            patch("app.api.routes.extract_from_bytes", return_value=_MOCK_OCR_RESULT),
            patch("app.api.routes.predict", return_value=_MOCK_PREDICTION),
            patch("app.api.routes.get_advice", new=AsyncMock(return_value=_MOCK_ADVICE)),
        ]
        if extra_patches:
            patches.extend(extra_patches)
        with patches[0], patches[1], patches[2]:
            return client.post(
                "/predict",
                files={"file": ("screen.png", _make_png_bytes(), "image/png")},
            )

    def test_returns_200(self, client):
        r = self._post(client)
        assert r.status_code == 200

    def test_label_in_response(self, client):
        r = self._post(client)
        assert r.json()["label"] == "Moderate"

    def test_probabilities_sum_to_one(self, client):
        r = self._post(client)
        proba = r.json()["probabilities"]
        assert abs(sum(proba.values()) - 1.0) < 1e-3

    def test_advice_present(self, client):
        r = self._post(client)
        assert "advice" in r.json()
        assert len(r.json()["advice"]) > 0

    def test_model_not_found_returns_503(self, client):
        with patch("app.api.routes.extract_from_bytes", return_value=_MOCK_OCR_RESULT):
            with patch(
                "app.api.routes.predict",
                side_effect=FileNotFoundError("model.pkl not found"),
            ):
                r = client.post(
                    "/predict",
                    files={"file": ("screen.png", _make_png_bytes(), "image/png")},
                )
        assert r.status_code == 503

    def test_missing_file_returns_422(self, client):
        r = client.post("/predict")
        assert r.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# POST /predict/manual
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictManualEndpoint:
    _valid_body = {
        "screen_time_hours": 6.0,
        "social_media_hours": 3.0,
        "gaming_hours": 1.0,
        "unlock_count": 90,
        "night_usage": 1,
    }

    def test_returns_200(self, client):
        with patch("app.api.routes.predict", return_value=_MOCK_PREDICTION):
            with patch("app.api.routes.get_advice", new=AsyncMock(return_value=_MOCK_ADVICE)):
                r = client.post("/predict/manual", json=self._valid_body)
        assert r.status_code == 200

    def test_negative_screen_time_rejected(self, client):
        body = dict(self._valid_body)
        body["screen_time_hours"] = -1
        r = client.post("/predict/manual", json=body)
        assert r.status_code == 422

    def test_over_24h_rejected(self, client):
        body = dict(self._valid_body)
        body["screen_time_hours"] = 25
        r = client.post("/predict/manual", json=body)
        assert r.status_code == 422

    def test_missing_required_field_rejected(self, client):
        body = {"social_media_hours": 2.0}
        r = client.post("/predict/manual", json=body)
        assert r.status_code == 422
