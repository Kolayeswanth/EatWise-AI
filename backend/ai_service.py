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

INGREDIENT_START_PATTERNS = (
    r"ingredients?",
    r"composition",
    r"склад",
    r"состав",
)

INGREDIENT_STOP_PATTERNS = (
    r"may contain",
    r"contains",
    r"nutrition",
    r"nutritional information",
    r"allergen",
    r"warning",
)

NOISE_TOKENS = {
    "fer",
    "who",
    "mac",
    "nan",
    "ingred",
    "bahi",
    "ingredient",
    "ingredients",
    "composition",
    "contains",
    "may",
    "contain",
    "nutrition",
    "nutritional",
    "information",
    "warning",
}

KEEP_PHRASES = {
    "cocoa butter",
}

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
    source_lines = lines or [part for part in re.split(r"\r?\n", raw_text) if part is not None]
    if not source_lines:
        return ""

    collected: List[str] = []
    started = False

    for raw_line in source_lines:
        line = str(raw_line).strip()
        if not line:
            if started and collected:
                break
            continue

        lowered = line.lower()
        if not started:
            if not any(re.search(rf"\b{pattern}\b", lowered, flags=re.IGNORECASE) for pattern in INGREDIENT_START_PATTERNS):
                continue
            started = True
            for pattern in INGREDIENT_START_PATTERNS:
                match = re.search(rf"{pattern}\s*[:\-]?\s*(.*)$", line, flags=re.IGNORECASE)
                if match:
                    remainder = match.group(1).strip()
                    if remainder:
                        collected.append(remainder)
                    break
            continue

        if any(re.search(rf"\b{pattern}\b", lowered, flags=re.IGNORECASE) for pattern in INGREDIENT_STOP_PATTERNS):
            break

        collected.append(line)

    section = "\n".join(collected).strip()
    if section:
        return section

    text = "\n".join(str(item).strip() for item in source_lines if str(item).strip())
    match = re.search(
        r"(?:ingredients?|composition|склад|состав)\s*[:\-]?\s*(.*?)(?:\n\s*\n|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _clean_ingredient_token(token: str) -> str:
    token = token.lower().strip()
    token = token.replace("/", " ")
    token = re.sub(r"\b\d+(?:[.,]\d+)?%?\b", " ", token)
    token = re.sub(r"[^a-z\s\-]", " ", token)
    token = re.sub(r"\s+", " ", token).strip(" .,-")

    if not token:
        return ""

    if len(token) < 3:
        return ""

    if token in NOISE_TOKENS:
        return ""

    words = [word for word in token.split() if len(word) >= 3 and word not in NOISE_TOKENS]
    if not words:
        return ""

    token = " ".join(words)
    if len(token.split()) > 6:
        return ""

    if token in {"ukrainian", "russian"}:
        return ""

    return token


def _normalize_ingredient_token(token: str) -> str:
    normalized = token.lower().strip()

    replacements = {
        "lecithins soya": "lecithin",
        "lecithin soya": "lecithin",
        "lecithins soy": "lecithin",
        "soy lecithin": "lecithin",
        "soya lecithin": "lecithin",
        "milk powder": "milk",
        "wheat flour": "wheat",
        "casein": "milk protein",
        "whey": "milk protein",
        "sodium benzoate": "preservative",
        "potassium sorbate": "preservative",
        "benzoate": "preservative",
    }

    for key, value in replacements.items():
        if key in normalized:
            normalized = value
            break

    if normalized in KEEP_PHRASES:
        return normalized

    if normalized.endswith("s") and len(normalized) > 4 and normalized not in {"cocoa butter"}:
        singular = normalized[:-1]
        if singular in KEEP_PHRASES:
            return singular

    return normalized


def extract_ingredients_from_ocr_text(raw_text: str, lines: Optional[List[str]] = None) -> List[str]:
    section = _extract_ingredient_section(raw_text, lines)
    if not section.strip():
        return []

    normalized = section.lower().replace(";", ",")
    normalized = re.sub(r"lecithins?\s*\(\s*soya\s*\)", "lecithin", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"lecithins?\s*\(\s*soy\s*\)", "lecithin", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"lecithins?\s+soya", "lecithin", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"lecithins?\s+soy", "lecithin", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\n+", ",", normalized)
    parts = re.split(r"[,\(\)\[\]{}]", normalized)

    cleaned: List[str] = []
    seen = set()

    for part in parts:
        token = _clean_ingredient_token(part)
        if not token:
            continue

        if any(re.search(rf"\b{pattern}\b", token, flags=re.IGNORECASE) for pattern in INGREDIENT_STOP_PATTERNS):
            break

        token = _normalize_ingredient_token(token)
        token = token.strip()

        if not token or len(token) < 3:
            continue

        if any(char.isdigit() for char in token):
            continue

        if not re.fullmatch(r"[a-z\-\s]+", token):
            continue

        if token in NOISE_TOKENS:
            continue

        normalized_token = re.sub(r"\s+", " ", token).strip()
        if normalized_token in seen:
            continue

        seen.add(normalized_token)
        cleaned.append(normalized_token)

        if len(cleaned) >= 30:
            break

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
        base = _normalize_ingredient_token(token.lower())
        for code, name in E_NUMBER_MAP.items():
            if code in base:
                base = base.replace(code, name)
                hidden.append(name)

        normalized_name = base
        if normalized_name == "preservative":
            hidden.append("preservative")
        if normalized_name == "milk protein":
            hidden.append("milk protein")

        normalized_name = normalized_name.strip(" .")
        if normalized_name and normalized_name not in normalized and len(normalized) < 30:
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
    logger.info("Rule-based ingredient normalization completed")
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
