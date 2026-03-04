"""
FastAPI route definitions.

Endpoints
---------
GET  /health        – liveness check
GET  /metrics       – model training metrics
POST /upload        – accept image, run OCR, return structured features
POST /predict       – accept image OR features, return full prediction + advice
"""
from __future__ import annotations

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.config import settings
from app.llm.advisor import get_advice
from app.ml.predictor import load_metrics, predict
from app.ocr.extractor import extract_from_bytes

logger = logging.getLogger(__name__)

router = APIRouter()

_MAX_BYTES = settings.max_upload_size_mb * 1024 * 1024


# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class OCRResponse(BaseModel):
    screen_time_hours: float
    social_media_hours: float
    gaming_hours: float
    unlock_count: int
    notification_count: int
    night_usage: int
    raw_text: str
    confidence: float


class PredictionResponse(BaseModel):
    label: str
    probabilities: dict[str, float]
    addiction_score: float
    features: dict[str, float]
    advice: str
    advice_source: str


class ManualPredictRequest(BaseModel):
    screen_time_hours: float = Field(..., ge=0, le=24)
    social_media_hours: float = Field(..., ge=0, le=24)
    gaming_hours: float = Field(0.0, ge=0, le=24)
    unlock_count: int = Field(..., ge=0, le=1000)
    night_usage: int = Field(0, ge=0, le=1)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

async def _read_and_validate_upload(file: UploadFile) -> bytes:
    """Read upload bytes, validate size and content type."""
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Uploaded file must be an image. Got: {content_type}",
        )
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )
    if len(data) > _MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_size_mb} MB limit.",
        )
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Liveness probe – always returns 200 when the server is up."""
    return HealthResponse(status="ok", version=settings.app_version)


@router.get("/metrics", tags=["System"])
async def get_model_metrics() -> dict:
    """Return the saved model training metrics."""
    metrics = load_metrics()
    if "error" in metrics:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=metrics["error"],
        )
    return metrics


@router.post("/upload", response_model=OCRResponse, tags=["OCR"])
async def upload_screenshot(
    file: Annotated[UploadFile, File(description="Smartphone screenshot (JPEG / PNG)")],
) -> OCRResponse:
    """
    Accept a smartphone screenshot, run EasyOCR, and return structured
    behavioural data extracted from the image.
    """
    data = await _read_and_validate_upload(file)
    try:
        result = extract_from_bytes(data)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected OCR error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OCR processing failed.",
        ) from exc

    return OCRResponse(**result)


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_from_screenshot(
    file: Annotated[UploadFile, File(description="Smartphone screenshot")],
) -> PredictionResponse:
    """
    Full pipeline: OCR → feature extraction → ML prediction → LLM advice.

    Returns addiction level, probabilities, score, and personalised advice.
    """
    data = await _read_and_validate_upload(file)

    # 1. OCR
    try:
        ocr_result = extract_from_bytes(data)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("OCR failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OCR processing failed.",
        ) from exc

    # 2. ML prediction
    try:
        pred = predict(ocr_result)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed.",
        ) from exc

    # 3. LLM advice
    advice_result = await get_advice(
        label=pred["label"],
        features=pred["features"],
        score=pred["addiction_score"],
    )

    return PredictionResponse(
        label=pred["label"],
        probabilities=pred["probabilities"],
        addiction_score=pred["addiction_score"],
        features=pred["features"],
        advice=advice_result["advice"],
        advice_source=advice_result["source"],
    )


@router.post("/predict/manual", response_model=PredictionResponse, tags=["Prediction"])
async def predict_manual(body: ManualPredictRequest) -> PredictionResponse:
    """
    Accept explicit numeric features (no screenshot required) and return
    the full prediction + advice.  Useful for testing and direct API calls.
    """
    features = body.model_dump()
    try:
        pred = predict(features)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    advice_result = await get_advice(
        label=pred["label"],
        features=pred["features"],
        score=pred["addiction_score"],
    )

    return PredictionResponse(
        label=pred["label"],
        probabilities=pred["probabilities"],
        addiction_score=pred["addiction_score"],
        features=pred["features"],
        advice=advice_result["advice"],
        advice_source=advice_result["source"],
    )
