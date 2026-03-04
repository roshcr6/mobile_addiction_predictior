"""
OCR extraction module.

Uses EasyOCR to extract text from a smartphone screenshot, then applies
regex patterns to parse structured behavioural data.

Key design decision:
    EasyOCR stitches all detected text into ONE flat string (no newlines).
    e.g. "Instagram 4h 10m WhatsApp 54m Battlegrounds 5h 47m"
    All parsers therefore search for APP-NAME immediately followed by a
    time value within the same flat string, rather than line-splitting.
"""
from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Lazy EasyOCR reader
# ──────────────────────────────────────────────────────────────────────────────
_reader = None


def _get_reader():
    """Return a cached EasyOCR reader (English)."""
    global _reader
    if _reader is None:
        import easyocr
        logger.info("Initialising EasyOCR reader...")
        _reader = easyocr.Reader(["en"], gpu=False)
        logger.info("EasyOCR reader ready.")
    return _reader


# ──────────────────────────────────────────────────────────────────────────────
# OCR error correction
# ──────────────────────────────────────────────────────────────────────────────

def _fix_ocr_errors(text: str) -> str:
    """Fix common OCR misreads: 'Sh 47m' -> '5h 47m', '1Om' -> '10m'."""
    text = re.sub(r'\bS(\s*h\b)', r'5\1', text)
    text = re.sub(r'\bI(\s*h\b)', r'1\1', text)
    text = re.sub(r'\bI(\s*m\b)', r'1\1', text)
    text = re.sub(r'(\d)O(m\b)', r'\g<1>0\2', text)
    text = re.sub(r'(\d)O(\d)', r'\g<1>0\2', text)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Time pattern helpers
# ──────────────────────────────────────────────────────────────────────────────

# Unlock count — two formats:
#   iOS  : keyword BEFORE number  e.g. "Pickups 87"  "Unlocks 132"
#   Android: number BEFORE keyword  e.g. "87 times"   "45 times opened"
# Notifications are explicitly excluded so they never corrupt unlock counts.
_UNLOCK_KW_BEFORE = re.compile(
    # iOS labels that appear BEFORE the count: "Pickups 87", "Unlocks 87"
    # "Times Opened" is excluded here — on Android the number always comes first
    r"(?:pickups?|unlocks?|phone\s*pickups?|phone\s*checks?)"
    r"\s{0,5}(\d{1,4})\b",
    re.I,
)
_UNLOCK_NUM_BEFORE = re.compile(
    # Android / generic: "64 times", "64 times opened", "87 unlocks"
    # No negative lookahead — "87 unlocks 34 notifications" must still capture 87
    r"\b(\d{1,4})\s*(?:times?(?:\s+opened)?|unlocks?|pickups?|phone\s*checks?)",
    re.I,
)

# Notifications — kept completely separate from unlocks
_NOTIF_KW_BEFORE = re.compile(
    r"\bnotifications?\s{0,5}(\d{1,4})\b",
    re.I,
)
_NOTIF_NUM_BEFORE = re.compile(
    r"\b(\d{1,4})\s*notifications?\b",
    re.I,
)

_NIGHT_MARKERS = re.compile(
    r"\b(2[2-3]|0[0-2]):[0-5]\d\b"
    r"|\b(?:midnight|late\s*night|night\s*usage|night\s*mode)\b",
    re.I,
)

# Time value patterns (used standalone)
_T_HM  = re.compile(r"(\d{1,2})\s*h(?:rs?|ours?)?\s*[,]?\s*(\d{1,2})\s*m(?:ins?|inutes?)?", re.I)
_T_H   = re.compile(r"(\d{1,2}(?:\.\d)?)\s*h(?:rs?|ours?)", re.I)
_T_M   = re.compile(r"\b(\d{1,3})\s*m(?:ins?|inutes?)?\b", re.I)
_T_CLN = re.compile(r"\b(\d{1,2}):(\d{2})\b")

# ── App name lists ────────────────────────────────────────────────────────────

_SOCIAL_NAMES = [
    "instagram", "facebook", "tiktok", "twitter", "snapchat",
    "youtube", "whatsapp", "telegram", "reddit", "pinterest",
    r"x\.com", "linkedin", "threads", "discord", "wechat",
]

_GAMING_NAMES = [
    "pubg", "battlegrounds", "fortnite", "roblox", "minecraft",
    r"clash\s+royale", r"clash\s+of\s+clans", r"brawl\s+stars",
    "freefire", r"free\s+fire", r"call\s+of\s+duty", r"\bcod\b",
    r"mobile\s+legends", "genshin", r"candy\s+crush", r"among\s+us",
    "pokemon", r"\bgames?\b",
]

# Category-level iOS Screen Time patterns
_CAT_SOCIAL = re.compile(
    r"(?:social(?:\s+networking)?|communication)\s+"
    r"(\d{1,2})\s*h(?:rs?)?\s*(\d{1,2})\s*m"
    r"|(?:social(?:\s+networking)?|communication)\s+(\d{1,2})\s*h(?:rs?)?(?!\d)"
    r"|(?:social(?:\s+networking)?|communication)\s+(\d{1,3})\s*m(?:ins?)?",
    re.I,
)
_CAT_GAMING = re.compile(
    r"(?:games?|gaming)\s+"
    r"(\d{1,2})\s*h(?:rs?)?\s*(\d{1,2})\s*m"
    r"|(?:games?|gaming)\s+(\d{1,2})\s*h(?:rs?)?(?!\d)"
    r"|(?:games?|gaming)\s+(\d{1,3})\s*m(?:ins?)?",
    re.I,
)


def _parse_hm_groups(groups) -> float:
    """Turn (h, m) or (h,) or (m,) groups from a regex match into hours."""
    vals = [g for g in groups if g is not None]
    if len(vals) >= 2:
        return round(int(vals[0]) + int(vals[1]) / 60, 2)
    if len(vals) == 1:
        return round(float(vals[0]) / 60, 2)
    return 0.0


def _category_hours(text: str, pat: re.Pattern) -> float:
    """Extract the first category-level duration from pat."""
    m = pat.search(text)
    if not m:
        return 0.0
    groups = [g for g in m.groups() if g is not None]
    if len(groups) >= 2:
        return round(int(groups[0]) + int(groups[1]) / 60, 2)
    if len(groups) == 1:
        val = float(groups[0])
        # If the matched substring contains 'h', it is already hours
        if re.search(r'\dh', m.group(0), re.I):
            return round(val, 2)
        return round(val / 60, 2)
    return 0.0


def _sum_app_times(text: str, names: list) -> float:
    """
    For each known app name, find the time value that immediately follows
    it in the flat OCR string and sum them all.

    Pattern:  <app_name>  <optional 1-3 words>  <Xh Ym | X hrs Y mins | Xh | Xm>
    """
    total = 0.0
    for name in names:
        pat = re.compile(
            rf"(?:{name})"
            r"\s+(?:\w+\s+){0,3}"
            r"(\d{1,2})\s*h(?:rs?|ours?)?\s*[,]?\s*(\d{1,2})\s*m(?:ins?|inutes?)?"
            r"|"
            rf"(?:{name})"
            r"\s+(?:\w+\s+){0,3}"
            r"(\d{1,2})\s*h(?:rs?|ours?)?(?!\s*\d)"
            r"|"
            rf"(?:{name})"
            r"\s+(?:\w+\s+){0,3}"
            r"(\d{1,3})\s*m(?:ins?|inutes?)?(?!\s*\w)",
            re.I,
        )
        for m in pat.finditer(text):
            groups = [g for g in m.groups() if g is not None]
            if len(groups) >= 2:
                total += int(groups[0]) + int(groups[1]) / 60
            elif len(groups) == 1:
                val = float(groups[0])
                # Decide if hours or minutes from which branch matched
                if re.search(r'\dh', m.group(0), re.I):
                    total += val
                else:
                    total += val / 60
    return round(total, 2)


def _text_to_hours(text: str) -> Optional[float]:
    """Parse the first time duration in *text*, return decimal hours."""
    text = _fix_ocr_errors(text)
    m = _T_HM.search(text)
    if m:
        return round(int(m.group(1)) + int(m.group(2)) / 60, 2)
    m = _T_CLN.search(text)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        if h <= 23 and mn <= 59:
            return round(h + mn / 60, 2)
    m = _T_H.search(text)
    if m:
        return round(float(m.group(1)), 2)
    m = _T_M.search(text)
    if m:
        val = int(m.group(1))
        if val <= 720:
            return round(val / 60, 2)
    return None


def _detect_night_usage(full_text: str) -> int:
    return 1 if _NIGHT_MARKERS.search(full_text) else 0


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def extract_from_bytes(image_bytes: bytes) -> dict:
    """Run OCR on raw image bytes and return structured behavioural features."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot open image: {exc}") from exc

    img_array = np.array(img)
    reader = _get_reader()
    logger.debug("Running EasyOCR on image (size=%s)...", img.size)
    results = reader.readtext(img_array, detail=1)

    if not results:
        logger.warning("EasyOCR returned no text.")
        return _empty_result()

    all_text = " ".join(r[1] for r in results)
    avg_conf = round(sum(r[2] for r in results) / len(results), 3)
    logger.debug("OCR raw text: %s", all_text[:300])
    return _parse_text(all_text, avg_conf)


def extract_from_path(image_path) -> dict:
    """Read an image file from disk and run OCR."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return extract_from_bytes(path.read_bytes())


def _parse_text(full_text: str, confidence: float) -> dict:
    """
    Apply all regex parsers to the flat OCR string and return a clean dict.

    Strategy
    --------
    1. Screen time  -> 'Screen Time Xh Ym' heading; else largest time value
    2. Social media -> sum per-app times; fallback: iOS category total
    3. Gaming       -> sum per-app times; fallback: iOS category total
    4. Unlocks      -> 'N times/unlocks/pickups'
    5. Night usage  -> hour markers 22-02 or keywords
    """
    text = _fix_ocr_errors(full_text)

    # ── 1. Screen time ────────────────────────────────────────────────────
    screen_time = 0.0
    for pat in [
        re.compile(r"screen\s*time\s+(\d{1,2})\s*h(?:rs?)?\s*(\d{1,2})\s*m", re.I),
        re.compile(r"screen\s*time\s+(\d{1,2})\s*hrs?\s*,?\s*(\d{1,2})\s*mins?", re.I),
        re.compile(r"screen\s*time\s+(\d{1,2})\s+(\d{2})\s*(?:mins?)?", re.I),
        re.compile(r"screen\s*time\s+(\d{1,2}(?:\.\d)?)\s*h", re.I),
    ]:
        m = pat.search(text)
        if m:
            groups = [g for g in m.groups() if g is not None]
            if len(groups) >= 2:
                screen_time = round(int(groups[0]) + int(groups[1]) / 60, 2)
            else:
                screen_time = round(float(groups[0]), 2)
            if screen_time:
                break

    if screen_time == 0.0:
        candidates = []
        for m in _T_HM.finditer(text):
            val = int(m.group(1)) + int(m.group(2)) / 60
            if 0.25 <= val <= 24:
                candidates.append(round(val, 2))
        for m in _T_H.finditer(text):
            val = float(m.group(1))
            if 0.25 <= val <= 24:
                candidates.append(round(val, 2))
        if candidates:
            screen_time = max(candidates)

    # ── 2. Social media ────────────────────────────────────────────────────
    social_media = _sum_app_times(text, _SOCIAL_NAMES)
    if social_media == 0.0:
        social_media = _category_hours(text, _CAT_SOCIAL)
    if screen_time > 0:
        social_media = min(social_media, screen_time)

    # ── 3. Gaming ──────────────────────────────────────────────────────────
    gaming = _sum_app_times(text, _GAMING_NAMES)
    if gaming == 0.0:
        gaming = _category_hours(text, _CAT_GAMING)
    if screen_time > 0:
        gaming = min(gaming, screen_time)

    # ── 4. Unlock count ────────────────────────────────────────────────────
    # Try number-before FIRST ("87 unlocks", "64 times opened") — more precise.
    # Fall back to keyword-before ("Pickups 87") for iOS labels.
    unlock_count = 0
    m = _UNLOCK_NUM_BEFORE.search(text)
    if m:
        unlock_count = int(m.group(1))
    else:
        m = _UNLOCK_KW_BEFORE.search(text)
        if m:
            unlock_count = int(m.group(1))

    # ── 5. Notification count ──────────────────────────────────────────────
    notification_count = 0
    m = _NOTIF_KW_BEFORE.search(text)
    if m:
        notification_count = int(m.group(1))
    else:
        m = _NOTIF_NUM_BEFORE.search(text)
        if m:
            notification_count = int(m.group(1))

    # ── 6. Night usage ─────────────────────────────────────────────────────
    night_usage = _detect_night_usage(text)

    logger.info(
        "Parsed -> screen=%.2fh  social=%.2fh  gaming=%.2fh  "
        "unlocks=%d  notifications=%d  night=%d",
        screen_time, social_media, gaming, unlock_count, notification_count, night_usage,
    )

    return {
        "screen_time_hours": screen_time,
        "social_media_hours": round(social_media, 2),
        "gaming_hours": round(gaming, 2),
        "unlock_count": unlock_count,
        "notification_count": notification_count,
        "night_usage": night_usage,
        "raw_text": full_text,
        "confidence": confidence,
    }


def _empty_result() -> dict:
    return {
        "screen_time_hours": 0.0,
        "social_media_hours": 0.0,
        "gaming_hours": 0.0,
        "unlock_count": 0,
        "notification_count": 0,
        "night_usage": 0,
        "raw_text": "",
        "confidence": 0.0,
    }
