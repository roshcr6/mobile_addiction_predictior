"""
ML tests covering:
- Dataset generation shape and label distribution
- Feature engineering and score formula
- Model training (mocked fit) and accuracy threshold
- Predictor loading and inference
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────────

class TestAddictionScore:
    def test_zero_usage_zero_score(self):
        from app.ml.features import compute_addiction_score

        assert compute_addiction_score(0, 0, 0, 0) == 0.0

    def test_formula_correctness(self):
        from app.ml.features import compute_addiction_score

        # 0.4*4 + 0.3*2 + 0.2*(100/100) + 0.1*1 = 1.6+0.6+0.2+0.1 = 2.5
        score = compute_addiction_score(4.0, 2.0, 100, 1)
        assert score == pytest.approx(2.5, rel=1e-4)

    def test_high_usage_high_score(self):
        from app.ml.features import compute_addiction_score

        score = compute_addiction_score(12.0, 8.0, 400, 1)
        assert score > 5.0

    def test_score_is_float(self):
        from app.ml.features import compute_addiction_score

        s = compute_addiction_score(5, 3, 80, 0)
        assert isinstance(s, float)


class TestLabelFromScore:
    def test_low_below_3(self):
        from app.ml.features import label_from_score

        assert label_from_score(2.99) == "Low"

    def test_moderate_at_3(self):
        from app.ml.features import label_from_score

        assert label_from_score(3.0) == "Moderate"

    def test_moderate_below_5(self):
        from app.ml.features import label_from_score

        assert label_from_score(4.99) == "Moderate"

    def test_high_at_5(self):
        from app.ml.features import label_from_score

        assert label_from_score(5.0) == "High"

    def test_zero_is_low(self):
        from app.ml.features import label_from_score

        assert label_from_score(0.0) == "Low"


class TestBuildFeatureVector:
    def test_shape(self):
        from app.ml.features import build_feature_vector

        ocr = {
            "screen_time_hours": 5.0,
            "social_media_hours": 2.0,
            "gaming_hours": 1.0,
            "unlock_count": 80,
            "night_usage": 1,
        }
        X = build_feature_vector(ocr)
        assert X.shape == (1, 5)

    def test_missing_keys_default_zero(self):
        from app.ml.features import build_feature_vector

        X = build_feature_vector({})
        assert X.shape == (1, 5)
        assert np.all(X == 0)

    def test_values_correct(self):
        from app.ml.features import build_feature_vector

        ocr = dict(screen_time_hours=3, social_media_hours=1,
                   gaming_hours=0.5, unlock_count=50, night_usage=0)
        X = build_feature_vector(ocr)
        assert X[0, 0] == 3.0
        assert X[0, 3] == 50.0


# ──────────────────────────────────────────────────────────────────────────────
# Dataset generation
# ──────────────────────────────────────────────────────────────────────────────

class TestDatasetGeneration:
    def test_shape(self):
        from app.ml.dataset_generator import generate_dataset

        df = generate_dataset(n_samples=100)
        assert len(df) == 100

    def test_required_columns(self):
        from app.ml.dataset_generator import generate_dataset

        df = generate_dataset(n_samples=50)
        required = [
            "screen_time_hours",
            "social_media_hours",
            "gaming_hours",
            "unlock_count",
            "night_usage",
            "addiction_score",
            "label",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_labels_valid(self):
        from app.ml.dataset_generator import generate_dataset

        df = generate_dataset(n_samples=200)
        assert set(df["label"].unique()).issubset({"Low", "Moderate", "High"})

    def test_all_three_labels_present(self):
        from app.ml.dataset_generator import generate_dataset

        df = generate_dataset(n_samples=500)
        assert len(df["label"].unique()) == 3

    def test_no_negative_values(self):
        from app.ml.dataset_generator import generate_dataset

        df = generate_dataset(n_samples=200)
        for col in ["screen_time_hours", "social_media_hours", "gaming_hours",
                    "unlock_count", "night_usage", "addiction_score"]:
            assert df[col].min() >= 0, f"Negative values in {col}"

    def test_save_creates_file(self, tmp_path):
        from app.ml.dataset_generator import generate_dataset, save_dataset

        df = generate_dataset(n_samples=50)
        out = tmp_path / "test_data.csv"
        save_dataset(df, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_reproducible_with_seed(self):
        from app.ml.dataset_generator import generate_dataset

        df1 = generate_dataset(n_samples=100, seed=0)
        df2 = generate_dataset(n_samples=100, seed=0)
        assert df1.equals(df2)


# ──────────────────────────────────────────────────────────────────────────────
# Model training (uses a small in-memory dataset to stay fast)
# ──────────────────────────────────────────────────────────────────────────────

class TestModelTraining:
    @pytest.fixture(autouse=True)
    def _patch_settings(self, monkeypatch, tmp_path):
        """Redirect all file paths to a temp directory."""
        csv = tmp_path / "data.csv"
        model = tmp_path / "model.pkl"
        metrics = tmp_path / "metrics.json"

        import app.config as cfg

        monkeypatch.setattr(cfg.settings, "dataset_path", str(csv))
        monkeypatch.setattr(cfg.settings, "model_path", str(model))
        monkeypatch.setattr(cfg.settings, "metrics_path", str(metrics))
        monkeypatch.setattr(cfg.settings, "test_size", 0.2)
        monkeypatch.setattr(cfg.settings, "cv_folds", 3)

        # Also patch wherever imported
        import app.ml.trainer as trainer_mod
        import app.ml.predictor as predictor_mod

        monkeypatch.setattr(trainer_mod.settings, "dataset_path", str(csv))
        monkeypatch.setattr(trainer_mod.settings, "model_path", str(model))
        monkeypatch.setattr(trainer_mod.settings, "metrics_path", str(metrics))
        monkeypatch.setattr(trainer_mod.settings, "test_size", 0.2)
        monkeypatch.setattr(trainer_mod.settings, "cv_folds", 3)
        monkeypatch.setattr(predictor_mod.settings, "model_path", str(model))
        monkeypatch.setattr(predictor_mod.settings, "metrics_path", str(metrics))

        self._model_path = model
        self._metrics_path = metrics
        self._csv_path = csv

    def _generate_and_save(self, n=300):
        from app.ml.dataset_generator import generate_dataset, save_dataset

        df = generate_dataset(n_samples=n)
        save_dataset(df, self._csv_path)

    def test_train_creates_model_file(self):
        self._generate_and_save()
        from app.ml.trainer import train

        train()
        assert self._model_path.exists()

    def test_train_creates_metrics_file(self):
        self._generate_and_save()
        from app.ml.trainer import train

        train()
        assert self._metrics_path.exists()
        data = json.loads(self._metrics_path.read_text())
        assert "models" in data

    def test_model_accuracy_above_threshold(self):
        self._generate_and_save(400)
        from app.ml.trainer import train

        metrics = train()
        rf_acc = metrics["models"]["RandomForest"]["accuracy"]
        assert rf_acc >= 0.75, f"Accuracy {rf_acc:.4f} below 75% threshold"

    def test_predict_returns_label(self):
        self._generate_and_save()
        from app.ml.trainer import train

        train()

        import app.ml.predictor as pred_mod

        pred_mod._model = None  # reset cache
        result = pred_mod.predict({
            "screen_time_hours": 8.0,
            "social_media_hours": 3.0,
            "gaming_hours": 1.0,
            "unlock_count": 120,
            "night_usage": 1,
        })
        assert result["label"] in {"Low", "Moderate", "High"}
        assert "probabilities" in result
        assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-3

    def test_predict_raises_without_model(self, tmp_path):
        import app.ml.predictor as pred_mod
        import app.config as cfg

        pred_mod._model = None
        with pytest.raises(FileNotFoundError):
            pred_mod.load_model(str(tmp_path / "nonexistent.pkl"))
