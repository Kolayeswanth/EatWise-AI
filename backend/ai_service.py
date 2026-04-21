from __future__ import annotations

import json
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

_client: Optional[genai.Client] = None


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


def _fallback_normalize(raw_text: str, fallback_ingredients: List[str]) -> dict:
    source = fallback_ingredients or [raw_text]
    normalized: List[str] = []
    breakdown: List[dict] = []
    hidden: List[str] = []
    allergens: List[str] = []

    text = " ".join(source).lower()
    tokens = [token.strip() for token in re.split(r"[,;\n]", text) if token.strip()]
    if not tokens:
        tokens = [text.strip()] if text.strip() else []

    for token in tokens:
        base = token
        for code, name in E_NUMBER_MAP.items():
            if code in base:
                base = base.replace(code, name)
                hidden.append(name)
        replacements = {
            "casein": "milk protein",
            "whey": "milk protein",
            "lecithin": "lecithin",
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


def _call_gemini(prompt: str) -> str:
    client = _get_client()
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    text = getattr(response, "text", None)
    if text:
        return text
    return str(response)


def analyze_ingredients_with_ai(raw_text: str, fallback_ingredients: List[str]) -> dict:
    prompt = f"""
You are a food safety assistant.

Normalize the OCR text into structured ingredients and detect hidden ingredients and allergens.
Return ONLY valid JSON with this schema:
{{
  "ai_ingredients": ["string"],
  "hidden_ingredients": ["string"],
  "ai_allergens": ["string"],
  "ingredient_breakdown": [{{"original": "string", "normalized": "string"}}]
}}

OCR text:
{raw_text}

Fallback ingredients:
{fallback_ingredients}

Rules:
- Normalize ingredient names to human-friendly names.
- Expand additives like E322 or sodium benzoate into clearer names if possible.
- Detect hidden ingredients such as preservatives, emulsifiers, colorants, or protein sources.
- Detect allergens using food knowledge, not just exact string matching.
""".strip()

    try:
        text = _call_gemini(prompt)
        data = _extract_json(text)
        if not isinstance(data, dict):
            raise ValueError("Gemini response was not a JSON object")
        data.setdefault("ai_ingredients", fallback_ingredients)
        data.setdefault("hidden_ingredients", [])
        data.setdefault("ai_allergens", [])
        data.setdefault("ingredient_breakdown", [])
        return data
    except Exception:
        return _fallback_normalize(raw_text, fallback_ingredients)


def generate_explanation(prediction: Dict, features: Dict[str, float], ingredients: List[str]) -> str:
    prompt = f"""
You are a food safety advisor.
Explain the contamination risk in one short, simple sentence for a student demo.
Avoid technical terms. Mention the strongest reasons if relevant.
Return plain text only.

Prediction: {prediction}
Features: {features}
Ingredients: {ingredients}
""".strip()

    try:
        text = _call_gemini(prompt).strip()
        return text if text else _fallback_explanation(prediction, features)
    except Exception:
        return _fallback_explanation(prediction, features)


def generate_recommendations(prediction: Dict, features: Dict[str, float], ingredients: List[str]) -> List[str]:
    prompt = f"""
You are a food safety advisor.
Generate 3 short, practical recommendations for the user.
Return ONLY valid JSON in this format:
{{"recommendations": ["string", "string", "string"]}}

Prediction: {prediction}
Features: {features}
Ingredients: {ingredients}

Rules:
- Keep each recommendation short and actionable.
- Focus on storage, ingredient choice, and risk reduction.
""".strip()

    try:
        text = _call_gemini(prompt)
        data = _extract_json(text)
        recommendations = data.get("recommendations", [])
        if isinstance(recommendations, list) and recommendations:
            return [str(item).strip() for item in recommendations if str(item).strip()][:3]
    except Exception:
        pass

    return _fallback_recommendations(prediction, features, ingredients)


def _fallback_explanation(prediction: Dict, features: Dict[str, float]) -> str:
    return (
        f"Risk is {prediction.get('risk_classification', 'unknown').lower()} because the ingredient mix, hygiene level, "
        f"and environmental risk point in that direction."
    )


def _fallback_recommendations(prediction: Dict, features: Dict[str, float], ingredients: List[str]) -> List[str]:
    recommendations = [
        "Store food below 5°C when possible.",
        "Choose products with fewer preservatives and additives.",
        "Check allergen labels carefully before consuming.",
    ]

    if features.get("erf", 0) > 0.6:
        recommendations[0] = "Reduce heat and humidity exposure during storage."
    if prediction.get("risk_classification") == "High":
        recommendations[1] = "Avoid products with many preservatives or unclear ingredients."
    return recommendations[:3]
