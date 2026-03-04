"""
Unit tests for the OCR extraction module.

Tests cover:
- regex parsing helpers
- time conversion logic
- field extraction from synthesised OCR text
- corrupted / empty image handling (mocked)
"""
from __future__ import annotations

import io
import re

import pytest
from PIL import Image, ImageDraw, ImageFont

from app.ocr.extractor import (
    _detect_night_usage,
    _parse_text,
    _text_to_hours,
)


# ──────────────────────────────────────────────────────────────────────────────
# _text_to_hours
# ──────────────────────────────────────────────────────────────────────────────

class TestTextToHours:
    def test_hours_and_minutes(self):
        assert _text_to_hours("5h 30m") == pytest.approx(5.5, rel=1e-2)

    def test_hours_only(self):
        assert _text_to_hours("8h") == pytest.approx(8.0, rel=1e-2)

    def test_decimal_hours(self):
        assert _text_to_hours("3.5 hours") == pytest.approx(3.5, rel=1e-2)

    def test_minutes_only(self):
        assert _text_to_hours("90m") == pytest.approx(1.5, rel=1e-2)

    def test_colon_format(self):
        assert _text_to_hours("2:45") == pytest.approx(2.75, rel=1e-2)

    def test_no_match_returns_none(self):
        assert _text_to_hours("no time here") is None

    def test_verbose_format(self):
        assert _text_to_hours("4 hours 15 minutes") == pytest.approx(4.25, rel=1e-2)


# ──────────────────────────────────────────────────────────────────────────────
# _detect_night_usage
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectNightUsage:
    def test_late_hour_detected(self):
        assert _detect_night_usage("Last used at 23:15") == 1

    def test_midnight_keyword(self):
        assert _detect_night_usage("midnight usage detected") == 1

    def test_daytime_not_detected(self):
        assert _detect_night_usage("Used at 14:30") == 0

    def test_empty_text(self):
        assert _detect_night_usage("") == 0

    def test_early_morning_detected(self):
        assert _detect_night_usage("Active at 01:45") == 1


# ──────────────────────────────────────────────────────────────────────────────
# _parse_text  (integration of all parsers)
# ──────────────────────────────────────────────────────────────────────────────

class TestParseText:
    _sample_text = (
        "Screen Time\n"
        "Daily Average: 7h 30m\n"
        "Instagram: 2h 45m\n"
        "TikTok: 1h 10m\n"
        "PUBG: 45m\n"
        "Phone unlocked 87 times\n"
        "Last used at 23:00"
    )

    def test_screen_time_extracted(self):
        result = _parse_text(self._sample_text, 0.9)
        assert result["screen_time_hours"] == pytest.approx(7.5, rel=0.05)

    def test_social_media_extracted(self):
        result = _parse_text(self._sample_text, 0.9)
        # Instagram 2h45m + TikTok 1h10m = ~3.92 h
        assert result["social_media_hours"] > 3.0

    def test_gaming_extracted(self):
        result = _parse_text(self._sample_text, 0.9)
        assert result["gaming_hours"] > 0

    def test_unlock_count_extracted(self):
        result = _parse_text(self._sample_text, 0.9)
        assert result["unlock_count"] == 87

    def test_night_usage_detected(self):
        result = _parse_text(self._sample_text, 0.9)
        assert result["night_usage"] == 1

    def test_confidence_stored(self):
        result = _parse_text(self._sample_text, 0.85)
        assert result["confidence"] == 0.85

    def test_raw_text_stored(self):
        result = _parse_text(self._sample_text, 0.9)
        assert "Screen Time" in result["raw_text"]

    def test_empty_text_zeroes(self):
        result = _parse_text("", 0.0)
        assert result["screen_time_hours"] == 0.0
        assert result["unlock_count"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# extract_from_bytes  (uses a real synthesised image)
# ──────────────────────────────────────────────────────────────────────────────

def _make_dummy_image_bytes(text: str = "Screen Time 5h 30m") -> bytes:
    """Create a minimal white PNG with text for OCR testing."""
    img = Image.new("RGB", (400, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestExtractFromBytes:
    def test_invalid_bytes_raises(self):
        from app.ocr.extractor import extract_from_bytes

        with pytest.raises(ValueError, match="Cannot open image"):
            extract_from_bytes(b"not-an-image")

    def test_empty_bytes_raises(self):
        from app.ocr.extractor import extract_from_bytes

        with pytest.raises(ValueError):
            extract_from_bytes(b"")

    def test_valid_image_returns_dict(self, monkeypatch):
        """Mock the EasyOCR reader so we don't need GPU / big model."""
        from app.ocr import extractor as ext_mod

        class _FakeReader:
            def readtext(self, arr, detail=1):
                return [
                    (None, "Screen Time Daily Average: 6h 00m", 0.95),
                    (None, "Instagram: 2h 30m", 0.90),
                    (None, "Unlocked 75 times", 0.88),
                    (None, "Last used 23:45", 0.80),
                ]

        monkeypatch.setattr(ext_mod, "_reader", _FakeReader())

        img_bytes = _make_dummy_image_bytes()
        result = ext_mod.extract_from_bytes(img_bytes)

        assert "screen_time_hours" in result
        assert "unlock_count" in result
        assert result["night_usage"] == 1
        assert result["confidence"] > 0
