from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from google import genai

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
USE_GEMINI = os.getenv("USE_GEMINI", "true").lower() == "true"

logger = logging.getLogger(__name__)
_client: Optional[genai.Client] = None

FALLBACK_INGREDIENTS = ["ingredient detection unavailable"]

COMMON_ALLERGENS = [
    "milk",
    "egg",
    "peanut",
    "tree nut",
    "nut",
    "almond",
    "walnut",
    "cashew",
    "pistachio",
    "soy",
    "wheat",
    "gluten",
    "fish",
    "shellfish",
    "sesame",
    "casein",
    "whey",
    "lactose",
]

E_NUMBER_MAP = {
    "e322": "lecithin",
    "e220": "sulphur dioxide",
    "e211": "sodium benzoate",
    "e202": "potassium sorbate",
    "e331": "sodium citrate",
    "e621": "monosodium glutamate",
    "e160a": "beta-carotene",
}


def _get_client() -> genai.Client:
    global _client
    if not USE_GEMINI:
        raise RuntimeError("USE_GEMINI is disabled")
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY missing in environment or .env file")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _call_gemini(prompt: str) -> str:
    logger.info("Gemini call requested")
    client = _get_client()
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    text = getattr(response, "text", None)
    if text:
        return text
    return str(response)


def _extract_ingredient_section(raw_text: str, lines: Optional[List[str]] = None) -> str:
    text = "\n".join(lines) if lines else raw_text
    if not text.strip():
        return ""

    patterns = [
        r"ingredients?\s*[:\-]\s*(.+)",
        r"composition\s*[:\-]\s*(.+)",
        r"contains\s*[:\-]\s*(.+)",
        r"склад\s*[:\-]\s*(.+)",
        r"состав\s*[:\-]\s*(.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1)

    return text


def extract_ingredients_from_ocr_text(raw_text: str, lines: Optional[List[str]] = None) -> List[str]:
    section = _extract_ingredient_section(raw_text, lines)
    if not section.strip():
        return []

    normalized = section.replace(";", ",")
    parts = re.split(r"[,\n\(\)\[\]{}]", normalized)

    cleaned: List[str] = []
    seen = set()
    for part in parts:
        token = part.strip().lower()
        if not token:
            continue

        token = re.sub(r"\b\d+[\.,]?\d*%?\b", "", token)
        token = re.sub(r"[^a-zа-яіїєґ0-9\-\s]", "", token, flags=re.IGNORECASE)
        token = re.sub(r"\s+", " ", token).strip(" .-")

        if not token or len(token) < 2:
            continue
        if token in {
            "ingredients",
            "composition",
            "contains",
            "may contain",
            "trace",
            "warning",
        }:
            continue

        if token not in seen:
            cleaned.append(token)
            seen.add(token)

    return cleaned


def _rule_based_normalize(raw_text: str, fallback_ingredients: List[str]) -> dict:
    source = [item.strip() for item in fallback_ingredients if str(item).strip()]
    if not source:
        source = extract_ingredients_from_ocr_text(raw_text)
    if not source:
        source = FALLBACK_INGREDIENTS[:]

    normalized: List[str] = []
    breakdown: List[dict] = []
    hidden: List[str] = []
    allergens: List[str] = []

    text = " ".join(source).lower()

    for token in source:
        base = token.lower()
        for code, name in E_NUMBER_MAP.items():
            if code in base:
                base = base.replace(code, name)
                hidden.append(name)

        replacements = {
            "casein": "milk protein",
            "whey": "milk protein",
            "sodium benzoate": "preservative",
            "potassium sorbate": "preservative",
            "benzoate": "preservative",
        }

        normalized_name = base
        for key, value in replacements.items():
            if key in normalized_name:
                normalized_name = value
                if value == "preservative":
                    hidden.append(value)

        normalized_name = normalized_name.strip(" .")
        if normalized_name and normalized_name not in normalized:
            normalized.append(normalized_name)
        breakdown.append({"original": token, "normalized": normalized_name or token})

    for allergen in COMMON_ALLERGENS:
        if allergen in text and allergen not in allergens:
            allergens.append(allergen)

    return {
        "ai_ingredients": normalized or source,
        "hidden_ingredients": list(dict.fromkeys(hidden)),
        "ai_allergens": list(dict.fromkeys(allergens)),
        "ingredient_breakdown": breakdown,
        "explanation": "",
        "recommendations": [],
    }


def analyze_ingredients_with_ai(raw_text: str, fallback_ingredients: List[str]) -> dict:
    base = _rule_based_normalize(raw_text, fallback_ingredients)

    if not USE_GEMINI:
        logger.info("Gemini disabled; using rule-based ingredient normalization")
        return base

    prompt = f"""
You are a food safety assistant.
Normalize the ingredient list, identify hidden ingredients, and detect allergens.
Return ONLY valid JSON:
{{
  "ai_ingredients": ["string"],
  "hidden_ingredients": ["string"],
  "ai_allergens": ["string"],
  "ingredient_breakdown": [{{"original": "string", "normalized": "string"}}]
}}

OCR text:
{raw_text}

Rule-based ingredients:
{base.get("ai_ingredients", [])}
""".strip()

    try:
        text = _call_gemini(prompt)
        data = _extract_json(text)
        if not isinstance(data, dict):
            raise ValueError("Gemini response was not a JSON object")
        data.setdefault("ai_ingredients", base.get("ai_ingredients", []))
        data.setdefault("hidden_ingredients", base.get("hidden_ingredients", []))
        data.setdefault("ai_allergens", base.get("ai_allergens", []))
        data.setdefault("ingredient_breakdown", base.get("ingredient_breakdown", []))
        logger.info("Gemini ingredient normalization success")
        return data
    except Exception:
        logger.exception("Gemini ingredient normalization failed; using rule-based output")
        return base


def generate_explanation(prediction: Dict, features: Dict[str, float], ingredients: List[str]) -> str:
    if not USE_GEMINI:
        return _fallback_explanation(prediction, features)

    prompt = f"""
You are a food safety advisor.
Explain the contamination risk in one short, simple sentence.
Return plain text only.

Prediction: {prediction}
Features: {features}
Ingredients: {ingredients}
""".strip()

    try:
        text = _call_gemini(prompt).strip()
        return text if text else _fallback_explanation(prediction, features)
    except Exception:
        logger.exception("Gemini explanation failed; using fallback")
        return _fallback_explanation(prediction, features)


def generate_risk_reasoning(prediction: Dict, features: Dict[str, float], ingredients: List[str]) -> str:
    if not USE_GEMINI:
        return _fallback_risk_reasoning(prediction, features)

    prompt = f"""
You are a food safety analyst.
Give concise reasoning in 2 short lines.
Return plain text only.

Prediction: {prediction}
Features: {features}
Ingredients: {ingredients}
""".strip()

    try:
        text = _call_gemini(prompt).strip()
        return text if text else _fallback_risk_reasoning(prediction, features)
    except Exception:
        logger.exception("Gemini risk reasoning failed; using fallback")
        return _fallback_risk_reasoning(prediction, features)


def generate_recommendations(prediction: Dict, features: Dict[str, float], ingredients: List[str]) -> List[str]:
    if not USE_GEMINI:
        return _fallback_recommendations(prediction, features, ingredients)

    prompt = f"""
You are a food safety advisor.
Generate 3 short, practical recommendations for the user.
Return ONLY valid JSON:
{{"recommendations": ["string", "string", "string"]}}

Prediction: {prediction}
Features: {features}
Ingredients: {ingredients}
""".strip()

    try:
        text = _call_gemini(prompt)
        data = _extract_json(text)
        recommendations = data.get("recommendations", [])
        if isinstance(recommendations, list) and recommendations:
            return [str(item).strip() for item in recommendations if str(item).strip()][:3]
    except Exception:
        logger.exception("Gemini recommendations failed; using fallback")

    return _fallback_recommendations(prediction, features, ingredients)


def _fallback_explanation(prediction: Dict, features: Dict[str, float]) -> str:
    return (
        f"Risk is {prediction.get('risk_classification', 'unknown').lower()} because the ingredient mix, hygiene level, "
        f"and environmental risk point in that direction."
    )


def _fallback_risk_reasoning(prediction: Dict, features: Dict[str, float]) -> str:
    return (
        f"The model predicted {prediction.get('risk_classification', 'unknown')} risk based on ingredient profile and environmental signals. "
        f"Higher ERF ({features.get('erf', 0):.2f}) and ingredient risk ({features.get('ingredient_risk', 0):.2f}) were key contributors."
    )


def _fallback_recommendations(prediction: Dict, features: Dict[str, float], ingredients: List[str]) -> List[str]:
    recommendations = [
        "Store food below 5 C when possible.",
        "Choose products with fewer preservatives and additives.",
        "Check allergen labels carefully before consuming.",
    ]

    if features.get("erf", 0) > 0.6:
        recommendations[0] = "Reduce heat and humidity exposure during storage."
    if prediction.get("risk_classification") == "High":
        recommendations[1] = "Avoid products with many preservatives or unclear ingredients."
    return recommendations[:3]
