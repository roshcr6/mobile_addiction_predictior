"""
Synthetic dataset generator for smartphone addiction prediction.

Generates realistic correlated samples based on published screen-time
research and digital wellbeing studies.

Usage::

    python -m app.ml.dataset_generator          # writes to data/
    python -m app.ml.dataset_generator --n 2000 # custom size

"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import settings
from app.ml.features import compute_addiction_score, label_from_score

logger = logging.getLogger(__name__)

RNG_SEED = settings.random_state


# ──────────────────────────────────────────────────────────────────────────────
# Core generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_dataset(n_samples: int = 1200, seed: int = RNG_SEED) -> pd.DataFrame:
    """
    Generate a synthetic dataset with realistic behavioural correlations.

    Parameters
    ----------
    n_samples:
        Number of rows to generate.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns:
        screen_time_hours, social_media_hours, gaming_hours,
        unlock_count, night_usage, addiction_score, label
    """
    rng = np.random.default_rng(seed)

    # ── 1. Screen time (hours per day) ─────────────────────────────────────
    # Mixture of 3 user archetypes:
    #   Light users  (40 %) : μ=2.5 h, σ=0.8
    #   Moderate users (35%): μ=5.0 h, σ=1.2
    #   Heavy users   (25 %): μ=9.0 h, σ=1.5
    archetype = rng.choice(["light", "moderate", "heavy"],
                           size=n_samples,
                           p=[0.40, 0.35, 0.25])

    screen_time = np.where(
        archetype == "light",
        rng.normal(2.5, 0.8, n_samples),
        np.where(
            archetype == "moderate",
            rng.normal(5.0, 1.2, n_samples),
            rng.normal(9.0, 1.5, n_samples),
        ),
    ).clip(0.5, 16.0)

    # ── 2. Social-media hours (correlated with screen time) ────────────────
    social_fraction = rng.beta(2.5, 2.5, n_samples)          # 0–1
    social_media = (screen_time * social_fraction * 0.65).clip(0.0, 12.0)

    # ── 3. Gaming hours (partially correlated) ────────────────────────────
    gaming_fraction = rng.beta(1.5, 4.0, n_samples)           # right-skewed
    gaming = (screen_time * gaming_fraction * 0.35).clip(0.0, 8.0)

    # ── 4. Unlock count (higher screen time → more unlocks) ───────────────
    base_unlocks = (screen_time * 12 + rng.normal(0, 10, n_samples)).clip(5, 400)
    # Add weekend spike noise
    spike = rng.integers(0, 30, n_samples)
    unlock_count = (base_unlocks + spike).astype(int).clip(5, 400)

    # ── 5. Night usage (higher social media → more likely) ────────────────
    night_prob = (social_media / 12.0 * 0.7 + 0.05).clip(0.05, 0.95)
    night_usage = rng.binomial(1, night_prob, n_samples)

    # ── 6. Compute score & label ──────────────────────────────────────────
    addiction_score = np.array([
        compute_addiction_score(st, sm, uc, nu)
        for st, sm, uc, nu in zip(screen_time, social_media, unlock_count, night_usage)
    ])
    labels = [label_from_score(s) for s in addiction_score]

    df = pd.DataFrame(
        {
            "screen_time_hours": np.round(screen_time, 2),
            "social_media_hours": np.round(social_media, 2),
            "gaming_hours": np.round(gaming, 2),
            "unlock_count": unlock_count,
            "night_usage": night_usage,
            "addiction_score": np.round(addiction_score, 3),
            "label": labels,
        }
    )

    logger.info(
        "Generated %d samples. Label distribution:\n%s",
        len(df),
        df["label"].value_counts().to_string(),
    )
    return df


def save_dataset(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """
    Save DataFrame to CSV.

    Parameters
    ----------
    df:
        DataFrame to save.
    path:
        Destination CSV path.  Defaults to ``settings.dataset_path``.
    """
    out = Path(path or settings.dataset_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Dataset saved → %s", out)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI: python -m app.ml.dataset_generator [--n N] [--out PATH]."""
    parser = argparse.ArgumentParser(description="Generate synthetic dataset.")
    parser.add_argument("--n", type=int, default=1200, help="Number of samples")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    df = generate_dataset(n_samples=args.n)
    save_dataset(df, path=args.out)
    print(df.head())
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")


if __name__ == "__main__":
    main()
