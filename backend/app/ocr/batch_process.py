"""
Batch OCR processor for the 19 sample screenshots.

Runs every image in backend/data/*.jpeg through the EasyOCR extractor,
computes the addiction score/label, and saves results to:
  - backend/data/sample_ocr_results.csv   (all extracted features)
  - backend/data/sample_summary.txt       (human-readable summary)

Usage (from the backend/ directory, with venv active):
    python -m app.ocr.batch_process
    python -m app.ocr.batch_process --dir backend/data --pattern "*.jpeg"
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from app.ml.features import compute_addiction_score, label_from_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def process_all(
    data_dir: Path = DEFAULT_DATA_DIR,
    pattern: str = "*.jpeg",
) -> list[dict]:
    """
    Run OCR on every image matching *pattern* inside *data_dir*.

    Returns
    -------
    list of result dicts, one per image.
    """
    # Import here so EasyOCR is only loaded when this script runs
    from app.ocr.extractor import extract_from_bytes

    images = sorted(data_dir.glob(pattern))
    if not images:
        # Try jpg as well
        images = sorted(data_dir.glob("*.jpg"))

    if not images:
        logger.error("No images found in %s matching '%s'", data_dir, pattern)
        return []

    logger.info("Found %d image(s) in %s", len(images), data_dir)

    results: list[dict] = []

    for idx, img_path in enumerate(images, start=1):
        logger.info("[%d/%d] Processing: %s", idx, len(images), img_path.name)
        try:
            ocr = extract_from_bytes(img_path.read_bytes())
            score = compute_addiction_score(
                screen_time_hours=ocr["screen_time_hours"],
                social_media_hours=ocr["social_media_hours"],
                unlock_count=ocr["unlock_count"],
                night_usage=ocr["night_usage"],
            )
            label = label_from_score(score)

            row = {
                "filename": img_path.name,
                "screen_time_hours": ocr["screen_time_hours"],
                "social_media_hours": ocr["social_media_hours"],
                "gaming_hours": ocr["gaming_hours"],
                "unlock_count": ocr["unlock_count"],
                "night_usage": ocr["night_usage"],
                "ocr_confidence": ocr["confidence"],
                "addiction_score": round(score, 3),
                "label": label,
                "raw_text_preview": ocr["raw_text"][:120].replace("\n", " "),
            }
            results.append(row)

            print(
                f"  ✓ {img_path.name:<12} | score={score:.2f} | label={label:<8} "
                f"| screen={ocr['screen_time_hours']:.1f}h "
                f"| social={ocr['social_media_hours']:.1f}h "
                f"| unlocks={ocr['unlock_count']}"
            )

        except Exception as exc:
            logger.warning("Failed to process %s: %s", img_path.name, exc)
            results.append(
                {
                    "filename": img_path.name,
                    "screen_time_hours": None,
                    "social_media_hours": None,
                    "gaming_hours": None,
                    "unlock_count": None,
                    "night_usage": None,
                    "ocr_confidence": None,
                    "addiction_score": None,
                    "label": "ERROR",
                    "raw_text_preview": str(exc),
                }
            )

    return results


def save_results(results: list[dict], data_dir: Path) -> None:
    """Write CSV and summary text file."""
    if not results:
        return

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = data_dir / "sample_ocr_results.csv"
    fieldnames = list(results[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info("CSV saved → %s", csv_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_path = data_dir / "sample_summary.txt"
    processed = [r for r in results if r["label"] != "ERROR"]
    errors = [r for r in results if r["label"] == "ERROR"]

    label_counts: dict[str, int] = {}
    for r in processed:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    lines = [
        "=" * 60,
        "  Batch OCR Results – Sample Screenshots",
        "=" * 60,
        f"  Total images    : {len(results)}",
        f"  Processed OK    : {len(processed)}",
        f"  Errors          : {len(errors)}",
        "",
        "  Label distribution:",
    ]
    for lbl in ["Low", "Moderate", "High"]:
        count = label_counts.get(lbl, 0)
        bar = "█" * count
        lines.append(f"    {lbl:<10} {bar}  ({count})")

    if processed:
        avg_score = sum(r["addiction_score"] for r in processed) / len(processed)
        avg_screen = sum(r["screen_time_hours"] for r in processed) / len(processed)
        lines += [
            "",
            f"  Avg addiction score : {avg_score:.2f}",
            f"  Avg screen time     : {avg_screen:.2f} h",
        ]

    if errors:
        lines += ["", "  Errors:"]
        for e in errors:
            lines.append(f"    ✗ {e['filename']}: {e['raw_text_preview']}")

    lines.append("=" * 60)
    text = "\n".join(lines)
    summary_path.write_text(text, encoding="utf-8")
    print("\n" + text)
    logger.info("Summary saved → %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch OCR all sample screenshots.")
    parser.add_argument(
        "--dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing screenshot images",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jpeg",
        help="Glob pattern (default: *.jpeg)",
    )
    args = parser.parse_args()

    data_dir = Path(args.dir)
    if not data_dir.exists():
        print(f"[ERROR] Directory not found: {data_dir}")
        return

    print(f"\n🔍 Scanning {data_dir} for '{args.pattern}' …\n")
    results = process_all(data_dir, pattern=args.pattern)
    save_results(results, data_dir)


if __name__ == "__main__":
    main()
