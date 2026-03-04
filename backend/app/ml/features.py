"""
Feature engineering for smartphone addiction prediction.

Centralises the addiction-score formula, label assignment, and the
feature-vector builder so every component (trainer, predictor, tests)
uses exactly the same logic.
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Label thresholds ───────────────────────────────────────────────────────────
LOW_THRESHOLD: float = 3.0
HIGH_THRESHOLD: float = 5.0

AddictionLabel = Literal["Low", "Moderate", "High"]

FEATURE_COLUMNS = [
    "screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "unlock_count",
    "night_usage",
]


# ──────────────────────────────────────────────────────────────────────────────
# Score & Label
# ──────────────────────────────────────────────────────────────────────────────

def compute_addiction_score(
    screen_time_hours: float,
    social_media_hours: float,
    unlock_count: int,
    night_usage: int,
) -> float:
    """
    Compute a continuous addiction score in the range [0, ~10+].

    Formula
    -------
    score = 0.4 × screen_time + 0.3 × social_media
            + 0.2 × (unlocks / 100) + 0.1 × night_usage

    Where screen_time and social_media are in hours, unlock_count is raw
    count, and night_usage is binary (0/1).

    Parameters
    ----------
    screen_time_hours:
        Total daily screen time in decimal hours.
    social_media_hours:
        Time spent on social media apps in decimal hours.
    unlock_count:
        Number of phone unlocks / pick-ups per day.
    night_usage:
        Binary flag: 1 if the device was used after 22:00, else 0.

    Returns
    -------
    float
        Addiction score (higher → more addicted).
    """
    score = (
        0.4 * screen_time_hours
        + 0.3 * social_media_hours
        + 0.2 * (unlock_count / 100.0)
        + 0.1 * night_usage
    )
    return round(float(score), 4)


def label_from_score(score: float) -> AddictionLabel:
    """
    Map a continuous addiction score to a categorical label.

    Thresholds
    ----------
    score < 3.0  → "Low"
    3.0 ≤ score < 5.0  → "Moderate"
    score ≥ 5.0  → "High"
    """
    if score < LOW_THRESHOLD:
        return "Low"
    elif score < HIGH_THRESHOLD:
        return "Moderate"
    else:
        return "High"


# ──────────────────────────────────────────────────────────────────────────────
# Feature vector builder
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_vector(ocr_result: dict) -> np.ndarray:
    """
    Convert an OCR extraction result dict into a 1-D feature array.

    Parameters
    ----------
    ocr_result:
        Dict produced by :func:`app.ocr.extractor.extract_from_bytes`.
        Expected keys: ``screen_time_hours``, ``social_media_hours``,
        ``gaming_hours``, ``unlock_count``, ``night_usage``.

    Returns
    -------
    np.ndarray of shape (1, 5)
    """
    row = [
        float(ocr_result.get("screen_time_hours", 0.0)),
        float(ocr_result.get("social_media_hours", 0.0)),
        float(ocr_result.get("gaming_hours", 0.0)),
        float(ocr_result.get("unlock_count", 0)),
        float(ocr_result.get("night_usage", 0)),
    ]
    return np.array(row, dtype=float).reshape(1, -1)


def build_feature_dataframe(rows: list[dict]) -> pd.DataFrame:
    """
    Build a DataFrame of shape (n, 5) from a list of OCR result dicts.

    Useful for batch prediction or training data ingestion.
    """
    records = [
        {
            "screen_time_hours": float(r.get("screen_time_hours", 0.0)),
            "social_media_hours": float(r.get("social_media_hours", 0.0)),
            "gaming_hours": float(r.get("gaming_hours", 0.0)),
            "unlock_count": float(r.get("unlock_count", 0)),
            "night_usage": float(r.get("night_usage", 0)),
        }
        for r in rows
    ]
    return pd.DataFrame(records, columns=FEATURE_COLUMNS)


def manual_features(
    screen_time_hours: float,
    social_media_hours: float,
    gaming_hours: float,
    unlock_count: int,
    night_usage: int,
) -> dict:
    """
    Build a feature dict from manual input (for the /predict endpoint
    when called with explicit parameters, not a screenshot).

    Also computes and appends the addiction_score.
    """
    score = compute_addiction_score(
        screen_time_hours, social_media_hours, unlock_count, night_usage
    )
    return {
        "screen_time_hours": screen_time_hours,
        "social_media_hours": social_media_hours,
        "gaming_hours": gaming_hours,
        "unlock_count": unlock_count,
        "night_usage": night_usage,
        "addiction_score": score,
    }
