from __future__ import annotations

import logging
import os
import re
import time
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT", "").rstrip("/")
AZURE_VISION_KEY = os.getenv("AZURE_VISION_KEY", "")
OCR_MAX_ATTEMPTS = 5
OCR_POLL_SECONDS = 1
OCR_HTTP_TIMEOUT = 5


def _azure_read_url() -> str:
    if not AZURE_VISION_ENDPOINT:
        return ""
    return f"{AZURE_VISION_ENDPOINT}/vision/v3.2/read/analyze"


def _extract_lines_from_azure(payload: dict) -> List[str]:
    lines: List[str] = []
    analyze_result = payload.get("analyzeResult", {})
    read_results = analyze_result.get("readResults", [])
    for page in read_results:
        for line in page.get("lines", []):
            text = str(line.get("text", "")).strip()
            if text:
                lines.append(text)
    return lines


def _azure_ocr(image_bytes: bytes) -> Dict[str, object]:
    if not AZURE_VISION_ENDPOINT or not AZURE_VISION_KEY:
        raise RuntimeError("Azure OCR configuration missing")

    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_VISION_KEY,
        "Content-Type": "application/octet-stream",
    }

    response = requests.post(_azure_read_url(), headers=headers, data=image_bytes, timeout=OCR_HTTP_TIMEOUT)
    response.raise_for_status()

    operation_location = response.headers.get("operation-location", "")
    if not operation_location:
        raise RuntimeError("Azure OCR did not return operation-location")

    poll_headers = {"Ocp-Apim-Subscription-Key": AZURE_VISION_KEY}
    for attempt in range(OCR_MAX_ATTEMPTS):
        poll_response = requests.get(operation_location, headers=poll_headers, timeout=OCR_HTTP_TIMEOUT)
        poll_response.raise_for_status()
        payload = poll_response.json()
        status = str(payload.get("status", "")).lower()

        if status == "succeeded":
            lines = _extract_lines_from_azure(payload)
            raw_text = "\n".join(lines).strip()
            logger.info("OCR success source=azure lines=%s", len(lines))
            return {"raw_text": raw_text, "lines": lines, "source": "azure"}

        if status == "failed":
            raise RuntimeError("Azure OCR operation failed")

        if attempt < OCR_MAX_ATTEMPTS - 1:
            time.sleep(OCR_POLL_SECONDS)

    raise TimeoutError("Azure OCR polling timed out")


def _lightweight_local_parse(image_bytes: bytes) -> Dict[str, object]:
    candidates: List[str] = []
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            decoded = image_bytes.decode(encoding, errors="ignore")
        except Exception:
            continue
        lines = [line.strip() for line in decoded.splitlines() if line.strip()]
        if lines:
            candidates.extend(lines)

    merged = "\n".join(candidates)
    # Keep only likely text-like lines to avoid binary gibberish.
    text_lines = [
        line
        for line in merged.splitlines()
        if len(re.findall(r"[A-Za-zА-Яа-яІіЇїЄєҐґ]", line)) >= 3
    ]

    raw_text = "\n".join(text_lines).strip()
    if not raw_text:
        raw_text = "ingredient detection unavailable"

    logger.info("OCR fallback triggered source=fallback lines=%s", len(text_lines))
    return {"raw_text": raw_text, "lines": text_lines, "source": "fallback"}


def extract_ingredients_from_image(image_bytes: bytes) -> Dict[str, object]:
    try:
        return _azure_ocr(image_bytes)
    except Exception:
        logger.exception("OCR azure path failed; switching to fallback")
        return _lightweight_local_parse(image_bytes)
