from __future__ import annotations

import logging
import os
import re
import shutil
import warnings
from io import BytesIO
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from backend.ai_service import extract_ingredients_from_image_with_ai

logger = logging.getLogger(__name__)

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    from rapidocr import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False


_rapidocr_engine = None


FALLBACK_INGREDIENTS = ["ingredient detection unavailable"]
ENABLE_GEMINI_OCR_FALLBACK = os.getenv("ENABLE_GEMINI_OCR_FALLBACK", "false").lower() == "true"
TESSERACT_LANGS = os.getenv("TESSERACT_LANGS", "eng+ukr")


def _prepare_ocr_images(img: np.ndarray) -> List[np.ndarray]:
    """Build multiple OCR-ready variants to improve detection on dense food labels."""
    variants: List[np.ndarray] = []

    scale = 2
    resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)

    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    variants.extend([gray, denoised, sharpened, adaptive, otsu])
    return variants


def _ocr_with_tesseract(image: np.ndarray) -> str:
    """Run several Tesseract passes and keep the strongest text result."""
    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 11",
        "--oem 3 --psm 12",
        "--oem 3 --psm 3",
    ]

    best_text = ""
    best_score = 0

    for variant in _prepare_ocr_images(image):
        for config in configs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                candidate = pytesseract.image_to_string(variant, lang=TESSERACT_LANGS, config=config)

            cleaned = candidate.strip()
            score = len(cleaned)
            if score > best_score:
                best_score = score
                best_text = candidate

    return best_text


def _get_rapidocr_engine():
    global _rapidocr_engine
    if _rapidocr_engine is None and RAPIDOCR_AVAILABLE:
        _rapidocr_engine = RapidOCR()
    return _rapidocr_engine


def _ocr_with_rapidocr(image_bytes: bytes) -> str:
    """Use RapidOCR as a stronger open-source fallback when Tesseract misses the text."""
    engine = _get_rapidocr_engine()
    if engine is None:
        return ""

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        result = engine(image)

        if result is None:
            return ""

        txts = None
        if hasattr(result, "txts"):
            txts = result.txts
        elif isinstance(result, tuple) and result:
            first = result[0]
            if hasattr(first, "txts"):
                txts = first.txts
            elif isinstance(first, (list, tuple)) and len(first) >= 2:
                txts = [item[1] for item in first if isinstance(item, (list, tuple)) and len(item) >= 2]

        if not txts:
            return ""

        lines = [str(text).strip() for text in txts if str(text).strip()]
        return "\n".join(lines)
    except Exception:
        logger.exception("RapidOCR extraction failed")
        return ""


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
    rapidocr_text = _ocr_with_rapidocr(image_bytes)
    if rapidocr_text.strip():
        logger.info("RapidOCR extracted text: %s", rapidocr_text[:500])
        return rapidocr_text, clean_ingredients(rapidocr_text)

    if not ENABLE_GEMINI_OCR_FALLBACK:
        logger.info("Gemini OCR fallback is disabled; returning open-source OCR fallback response")
        if raw_text.strip():
            return raw_text, clean_ingredients(raw_text)
        return "ingredient detection unavailable", []

    ai_result = extract_ingredients_from_image_with_ai(image_bytes)
    ai_ingredients = ai_result.get("ingredients", [])
    ai_raw_text = ai_result.get("raw_text", "")

    logger.info(
        "OCR fallback: raw_text=%s ai_ingredients=%s",
        (ai_raw_text or raw_text)[:500],
        ai_ingredients,
    )

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

            raw_text = _ocr_with_tesseract(img)

            logger.info("OCR extracted text: %s", raw_text[:500])

            ingredients = clean_ingredients(raw_text)
            if ingredients:
                return raw_text, ingredients
        except Exception:
            logger.exception("OCR extraction failed")
            raw_text = ""

    return _fallback_to_ai(image_bytes, raw_text=raw_text)
