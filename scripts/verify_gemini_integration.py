import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ai_service import analyze_ingredients_with_ai

sample_text = "Ingredients: E322, Sodium Benzoate, Casein, Wheat Flour, Sugar"
sample_fallback = ["E322", "Sodium Benzoate", "Casein", "Wheat Flour", "Sugar"]

result = analyze_ingredients_with_ai(raw_text=sample_text, fallback_ingredients=sample_fallback)

print("Gemini structured ingredient output:")
print(json.dumps(result, indent=2, ensure_ascii=False))
