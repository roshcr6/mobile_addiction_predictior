"""
ML model trainer.

Trains a Logistic Regression and a Random Forest classifier on the
synthetic smartphone addiction dataset, performs cross-validation,
saves the best model (Random Forest) and detailed metrics to disk.

Usage::

    python -m app.ml.trainer          # trains with defaults
    python -m app.ml.trainer --force  # regenerates dataset first

"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from app.config import settings
from app.ml.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_training_data(path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load CSV dataset and return feature matrix X and label vector y.

    Parameters
    ----------
    path:
        CSV path.  Defaults to ``settings.dataset_path``.

    Returns
    -------
    X : np.ndarray, shape (n, 5)
    y : np.ndarray, shape (n,)  – str labels "Low" / "Moderate" / "High"
    """
    csv_path = Path(path or settings.dataset_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Run 'python -m app.ml.dataset_generator' first."
        )
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLUMNS].values.astype(float)
    y = df["label"].values
    logger.info("Loaded %d samples from %s", len(df), csv_path)
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(force_regen: bool = False) -> dict:
    """
    Full training pipeline.

    Steps:
    1. Optionally regenerate dataset.
    2. Train/test split.
    3. Train Logistic Regression + Random Forest.
    4. Cross-validate both.
    5. Evaluate on held-out test set.
    6. Save best model + metrics.

    Parameters
    ----------
    force_regen:
        If True, regenerate the dataset before training.

    Returns
    -------
    dict
        Full metrics report including both models.
    """
    if force_regen:
        from app.ml.dataset_generator import generate_dataset, save_dataset

        logger.info("Regenerating synthetic dataset…")
        save_dataset(generate_dataset())

    X, y = load_training_data()

    # ── Split ────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=y,
    )
    logger.info("Train: %d  Test: %d", len(X_train), len(X_test))

    cv = StratifiedKFold(
        n_splits=settings.cv_folds,
        shuffle=True,
        random_state=settings.random_state,
    )

    # ── Logistic Regression ───────────────────────────────────────────────
    lr = LogisticRegression(
        max_iter=1000,
        random_state=settings.random_state,
        class_weight="balanced",
    )
    lr_cv_scores = cross_val_score(lr, X_train, y_train, cv=cv, scoring="f1_macro")
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_metrics = _compute_metrics(y_test, lr_preds, "LogisticRegression", lr_cv_scores)
    logger.info("LR accuracy: %.4f  CV F1: %.4f", lr_metrics["accuracy"], lr_metrics["cv_f1_mean"])

    # ── Random Forest ─────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        random_state=settings.random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="f1_macro")
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_metrics = _compute_metrics(y_test, rf_preds, "RandomForest", rf_cv_scores)
    logger.info("RF accuracy: %.4f  CV F1: %.4f", rf_metrics["accuracy"], rf_metrics["cv_f1_mean"])

    # ── Feature importances (RF) ──────────────────────────────────────────
    importances = dict(zip(FEATURE_COLUMNS, rf.feature_importances_.round(4).tolist()))

    # ── Save best model (RF) ──────────────────────────────────────────────
    model_path = Path(settings.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, model_path)
    logger.info("Model saved → %s", model_path)

    # ── Save metrics ──────────────────────────────────────────────────────
    metrics = {
        "best_model": "RandomForest",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_importances": importances,
        "models": {
            "LogisticRegression": lr_metrics,
            "RandomForest": rf_metrics,
        },
    }
    metrics_path = Path(settings.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved → %s", metrics_path)

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    cv_scores: np.ndarray,
) -> dict:
    """Compute a comprehensive metrics dict for a single model."""
    labels = ["Low", "Moderate", "High"]
    return {
        "model": model_name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "cv_f1_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_std": round(float(cv_scores.std()), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train addiction prediction models.")
    parser.add_argument(
        "--force", action="store_true", help="Regenerate synthetic dataset first."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    metrics = train(force_regen=args.force)
    rf = metrics["models"]["RandomForest"]
    print(f"\n✅  Training complete.")
    print(f"   Random Forest accuracy : {rf['accuracy']:.4f}")
    print(f"   Random Forest CV F1    : {rf['cv_f1_mean']:.4f} ± {rf['cv_f1_std']:.4f}")


if __name__ == "__main__":
    main()
