from __future__ import annotations

import io
import re
import warnings
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False


def check_tesseract_availability() -> bool:
    """Check if Tesseract is available on the system."""
    if not PYTESSERACT_AVAILABLE:
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
    text = re.sub(r"ingredients?[:\\n]", "", text)
    text = text.replace(";", ",")
    text = re.sub(r"[^a-z0-9,()\- ]+", " ", text)

    chunks = [x.strip(" .") for x in text.split(",")]
    cleaned = []
    for chunk in chunks:
        if not chunk:
            continue
        if len(chunk) < 2:
            continue
        cleaned.append(chunk)

    seen = set()
    deduped = []
    for item in cleaned:
        if item not in seen:
            deduped.append(item)
            seen.add(item)

    return deduped


def extract_ingredients_from_image(image_bytes: bytes) -> Tuple[str, List[str]]:
    """
    Extract ingredients from image using pytesseract.
    
    Falls back gracefully if Tesseract is not available or OCR fails.
    """
    if not PYTESSERACT_AVAILABLE:
        fallback_msg = "OCR library not available. Please install pytesseract and Tesseract-OCR."
        return fallback_msg, []
    
    if not check_tesseract_availability():
        fallback_msg = "Tesseract-OCR is not installed on this system. OCR is unavailable."
        return fallback_msg, []
    
    try:
        # Read image from bytes using OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            fallback_msg = "Failed to decode image. Please provide a valid image file."
            return fallback_msg, []
        
        # Convert BGR to RGB for pytesseract
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Optional: Improve OCR accuracy with image preprocessing
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply thresholding to improve contrast
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Extract text using pytesseract
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_text = pytesseract.image_to_string(binary)
        
        if not raw_text or not raw_text.strip():
            fallback_msg = "No text detected in image. Image may be blank or unreadable."
            return fallback_msg, []
        
        # Clean and parse ingredients
        ingredients = clean_ingredients(raw_text)
        return raw_text, ingredients
        
    except Exception as e:
        fallback_msg = f"OCR processing failed: {str(e)}"
        return fallback_msg, []
