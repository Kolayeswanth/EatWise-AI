#!/usr/bin/env python3
"""Test script for OCR service migration."""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path.parent))

from backend.ocr_service import check_tesseract_availability, extract_ingredients_from_image


def test_availability():
    """Test if Tesseract is available."""
    print("Testing Tesseract availability...")
    available = check_tesseract_availability()
    print(f"  Tesseract available: {available}")
    return available


def test_empty_image():
    """Test with empty image bytes."""
    print("\nTesting with empty image bytes...")
    raw_text, ingredients = extract_ingredients_from_image(b"")
    print(f"  Raw text: {raw_text}")
    print(f"  Ingredients: {ingredients}")
    assert isinstance(raw_text, str), "raw_text should be string"
    assert isinstance(ingredients, list), "ingredients should be list"
    print("  ✓ Handled gracefully")


def test_invalid_image():
    """Test with invalid image data."""
    print("\nTesting with invalid image data...")
    raw_text, ingredients = extract_ingredients_from_image(b"not an image")
    print(f"  Raw text: {raw_text}")
    print(f"  Ingredients: {ingredients}")
    assert isinstance(raw_text, str), "raw_text should be string"
    assert isinstance(ingredients, list), "ingredients should be list"
    print("  ✓ Handled gracefully")


def test_error_handling():
    """Test error handling without crashing."""
    print("\nTesting error handling...")
    try:
        # Test various invalid inputs
        test_cases = [b"", b"invalid", None]
        
        for test_input in test_cases[:-1]:  # Skip None for now
            raw_text, ingredients = extract_ingredients_from_image(test_input)
            assert isinstance(raw_text, str), f"Failed for input {test_input}"
            assert isinstance(ingredients, list), f"Failed for input {test_input}"
        
        print("  ✓ All error cases handled without crashing")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        raise


if __name__ == "__main__":
    print("=" * 50)
    print("OCR Service Migration Test")
    print("=" * 50)
    
    test_availability()
    test_empty_image()
    test_invalid_image()
    test_error_handling()
    
    print("\n" + "=" * 50)
    print("✓ All tests passed!")
    print("=" * 50)
