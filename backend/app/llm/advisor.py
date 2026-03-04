"""
LLM Advisor module.

Tries to reach a locally running Ollama instance first.
Falls back to a deterministic rule-based engine when Ollama is
unavailable or times out.
"""
from __future__ import annotations

import logging
from typing import Literal

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

AddictionLabel = Literal["Low", "Moderate", "High"]

# ──────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a digital wellness coach. Based on the user's smartphone usage "
    "statistics, give specific, empathetic, and actionable advice to help "
    "them reduce addiction and improve their mental wellbeing. Keep your "
    "response under 200 words and use bullet points."
)


def _build_user_prompt(features: dict, label: str, score: float) -> str:
    return (
        f"My smartphone addiction level is: **{label}** (score {score:.2f}/10).\n\n"
        f"My daily usage data:\n"
        f"- Total screen time : {features.get('screen_time_hours', 0):.1f} hours\n"
        f"- Social media time : {features.get('social_media_hours', 0):.1f} hours\n"
        f"- Gaming time       : {features.get('gaming_hours', 0):.1f} hours\n"
        f"- Phone unlocks     : {features.get('unlock_count', 0)} times\n"
        f"- Night-time usage  : {'Yes' if features.get('night_usage') else 'No'}\n\n"
        "Please give me personalised advice."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Ollama client
# ──────────────────────────────────────────────────────────────────────────────

async def _call_ollama(prompt: str) -> str | None:
    """
    Send a chat request to a local Ollama instance.

    Returns the assistant message string, or ``None`` on any error.
    """
    url = f"{settings.ollama_base_url}/api/chat"
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]
    except Exception as exc:
        logger.warning("Ollama unavailable (%s) – using rule-based advisor.", exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Rule-based fallback
# ──────────────────────────────────────────────────────────────────────────────

_ADVICE: dict[str, list[str]] = {
    "Low": [
        "Great job maintaining a healthy digital balance!",
        "• Keep your daily screen time below 3 hours.",
        "• Schedule one full 'phone-free' hour each morning.",
        "• Continue avoiding phone use after 10 PM.",
        "• Turn on grayscale mode in the evenings to reduce engagement.",
        "• Review your app usage weekly to spot any creeping habits.",
    ],
    "Moderate": [
        "You're showing some patterns of over-reliance on your phone.",
        "• Enable Screen Time / Digital Wellbeing app limits for social media.",
        "• Try the 20-20-20 rule: every 20 min, look 20 ft away for 20 sec.",
        "• Keep your phone out of the bedroom – use a separate alarm clock.",
        "• Replace one daily scroll session with a 10-minute walk.",
        "• Disable non-essential push notifications to reduce unlock frequency.",
        "• Consider one tech-free day per week (e.g., Sunday afternoons).",
    ],
    "High": [
        "⚠️ Your usage indicates high smartphone dependency. Act now.",
        "• Set strict app limits: max 1 hour Social Media per day.",
        "• Install an app blocker (e.g., Freedom, Cold Turkey) for evenings.",
        "• Charge your phone outside the bedroom – night usage is disrupting sleep.",
        "• Practice the 'phone-in-pocket' rule: phone stays pocketed in social settings.",
        "• Try a 7-day phone detox challenge – reduce total use by 50 %.",
        "• Replace the first 30 min of your day (no phone) with journaling or exercise.",
        "• Consider speaking with a digital wellness counsellor or therapist.",
        "• Track your progress daily – small wins build lasting habits.",
    ],
}


def _rule_based_advice(label: AddictionLabel, features: dict) -> str:
    """Return deterministic advice based on label and key feature values."""
    lines = list(_ADVICE.get(label, _ADVICE["Moderate"]))

    # Personalise based on dominant factor
    st = features.get("screen_time_hours", 0)
    sm = features.get("social_media_hours", 0)
    nu = features.get("night_usage", 0)
    uc = features.get("unlock_count", 0)

    if sm > 3:
        lines.append("• Your social media use is particularly high – "
                     "try a 3-day social media detox.")
    if nu:
        lines.append("• Night-time usage can disrupt sleep cycles – "
                     "set a phone curfew at 9 PM.")
    if uc > 150:
        lines.append(f"• {uc} unlocks/day is very high – "
                     "try disabling most notifications.")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

async def get_advice(
    label: AddictionLabel,
    features: dict,
    score: float,
) -> dict:
    """
    Return personalised advice for the given prediction result.

    Tries Ollama first; falls back to the rule-based engine.

    Parameters
    ----------
    label:
        Predicted addiction label.
    features:
        Feature dict (same as predictor output).
    score:
        Addiction score float.

    Returns
    -------
    dict with keys ``advice`` (str) and ``source`` ("ollama" | "rule-based").
    """
    prompt = _build_user_prompt(features, label, score)
    llm_response = await _call_ollama(prompt)

    if llm_response:
        return {"advice": llm_response, "source": "ollama"}

    fallback = _rule_based_advice(label, features)
    return {"advice": fallback, "source": "rule-based"}
