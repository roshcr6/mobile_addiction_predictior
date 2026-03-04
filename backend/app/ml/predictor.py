"""
Model loading and inference module.

A single module-level ``_model`` instance is loaded once at startup
(or lazily on first call) so every request reuses the same object.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from app.config import settings
from app.ml.features import (
    FEATURE_COLUMNS,
    build_feature_vector,
    compute_addiction_score,
    label_from_score,
)

logger = logging.getLogger(__name__)

_model: Optional[RandomForestClassifier] = None


# ──────────────────────────────────────────────────────────────────────────────
# Model lifecycle
# ──────────────────────────────────────────────────────────────────────────────

def load_model(path: str | None = None) -> RandomForestClassifier:
    """
    Load the trained model from disk into the module-level cache.

    Parameters
    ----------
    path:
        Explicit path to the ``.pkl`` file.  Defaults to
        ``settings.model_path``.

    Returns
    -------
    Loaded sklearn estimator.

    Raises
    ------
    FileNotFoundError
        When the model file does not exist.
    """
    global _model
    model_path = Path(path or settings.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Run 'python -m app.ml.trainer' to train the model."
        )
    _model = joblib.load(model_path)
    logger.info("Model loaded from %s (type=%s)", model_path, type(_model).__name__)
    return _model


def get_model() -> RandomForestClassifier:
    """Return the cached model, loading it lazily if necessary."""
    global _model
    if _model is None:
        _model = load_model()
    return _model


# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────

def predict(features: dict) -> dict:
    """
    Run inference given a feature dict.

    Parameters
    ----------
    features:
        Dict with keys matching ``FEATURE_COLUMNS`` (from OCR extraction
        or manual input).  Extra keys are silently ignored.

    Returns
    -------
    dict with keys:
        ``label``            – "Low" / "Moderate" / "High"
        ``probabilities``    – {label: probability, …}
        ``addiction_score``  – float
        ``features``         – sanitised feature dict
    """
    model = get_model()

    X = build_feature_vector(features)
    label: str = model.predict(X)[0]
    proba: np.ndarray = model.predict_proba(X)[0]
    classes: list[str] = list(model.classes_)

    score = compute_addiction_score(
        screen_time_hours=float(features.get("screen_time_hours", 0)),
        social_media_hours=float(features.get("social_media_hours", 0)),
        unlock_count=int(features.get("unlock_count", 0)),
        night_usage=int(features.get("night_usage", 0)),
    )

    probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

    result = {
        "label": label,
        "probabilities": probabilities,
        "addiction_score": round(score, 3),
        "features": {
            col: features.get(col, 0.0) for col in FEATURE_COLUMNS
        },
    }
    logger.debug("Prediction: %s (score=%.3f)", label, score)
    return result


def load_metrics(path: str | None = None) -> dict:
    """
    Read the saved training-metrics JSON.

    Returns
    -------
    dict or {"error": str} when file is missing.
    """
    metrics_path = Path(path or settings.metrics_path)
    if not metrics_path.exists():
        return {"error": "Metrics file not found. Train the model first."}
    return json.loads(metrics_path.read_text())
