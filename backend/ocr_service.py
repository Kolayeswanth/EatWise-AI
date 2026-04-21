from __future__ import annotations

import re
import shutil
import warnings
from typing import List, Tuple

import cv2
import numpy as np

from backend.ai_service import extract_ingredients_from_image_with_ai

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False


FALLBACK_INGREDIENTS = ["ingredient detection unavailable"]


def check_tesseract_availability() -> bool:
    """Check if Tesseract is available on the system."""
    if not PYTESSERACT_AVAILABLE:
        return False

    if shutil.which("tesseract") is None:
        return False

    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def clean_ingredients(raw_text: str) -> List[str]:
    """Clean and parse extracted text into ingredient list."""
    if not raw_text or not raw_text.strip():
        return []
    
    text = raw_text.lower()
    text = re.sub(r"\bingredients?\b\s*[:;]?", "", text)
    text = text.replace(";", ",")
    text = re.sub(r"[^a-z0-9,()\- ]+", " ", text)

    chunks = [x.strip(" .") for x in text.split(",")]
    cleaned = []
    for chunk in chunks:
        if not chunk:
            continue
        if len(chunk) < 2:
            continue
        if chunk in {"ingredients", "contains"}:
            continue
        cleaned.append(chunk)

    seen = set()
    deduped = []
    for item in cleaned:
        if item not in seen:
            deduped.append(item)
            seen.add(item)

    return deduped


def _fallback_to_ai(image_bytes: bytes, raw_text: str = "") -> Tuple[str, List[str]]:
    ai_result = extract_ingredients_from_image_with_ai(image_bytes)
    ai_ingredients = ai_result.get("ingredients", [])
    ai_raw_text = ai_result.get("raw_text", "")

    if ai_ingredients:
        return ai_raw_text or raw_text or ", ".join(ai_ingredients), ai_ingredients

    if raw_text.strip():
        return raw_text, clean_ingredients(raw_text)

    return "Unable to extract ingredients from the image.", FALLBACK_INGREDIENTS


def extract_ingredients_from_image(image_bytes: bytes) -> Tuple[str, List[str]]:
    """
    Extract ingredients from image using pytesseract first.

    If Tesseract is unavailable or returns empty results, Gemini image extraction is used as a fallback.
    """
    raw_text = ""

    if PYTESSERACT_AVAILABLE and check_tesseract_availability():
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return _fallback_to_ai(image_bytes)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_text = pytesseract.image_to_string(binary)

            ingredients = clean_ingredients(raw_text)
            if ingredients:
                return raw_text, ingredients
        except Exception:
            raw_text = ""

    return _fallback_to_ai(image_bytes, raw_text=raw_text)
