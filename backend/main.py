from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.ai_service import analyze_ingredients_with_ai, generate_explanation, generate_recommendations, generate_risk_reasoning
from backend.ml_service import HybridRiskModel
from backend.ocr_service import extract_ingredients_from_image
from backend.schemas import AnalyzeImageResponse, ExplainResponse, PredictRequest, PredictResponse

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)

app = FastAPI(title="Food Safety Risk Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = HybridRiskModel()


@app.get("/")
def health() -> dict:
    return {"status": "ok", "service": "food-safety-api"}


@app.post("/analyze-image", response_model=AnalyzeImageResponse)
async def analyze_image(file: UploadFile = File(...)) -> AnalyzeImageResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image file.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image file.")

    raw_text, ingredients = extract_ingredients_from_image(content)

    if ingredients:
        ai_result = analyze_ingredients_with_ai(raw_text=raw_text, fallback_ingredients=ingredients)
    else:
        ai_result = {
            "ai_ingredients": [],
            "hidden_ingredients": [],
            "ai_allergens": [],
            "ingredient_breakdown": [],
        }
    ai_ingredients = ai_result.get("ai_ingredients", ingredients) or ingredients
    hidden_ingredients = ai_result.get("hidden_ingredients", [])
    ai_allergens = ai_result.get("ai_allergens", [])
    ingredient_breakdown = ai_result.get("ingredient_breakdown", [])

    logger.info(
        "Analyze image input: raw_text=%s ingredients=%s ai_result=%s",
        raw_text[:500],
        ingredients,
        {
            "ai_ingredients": ai_ingredients,
            "hidden_ingredients": hidden_ingredients,
            "ai_allergens": ai_allergens,
        },
    )

    allergens = model.detect_allergens(ai_ingredients or ingredients)
    merged_allergens = list(dict.fromkeys(allergens + [item for item in ai_allergens if item not in allergens]))

    return AnalyzeImageResponse(
        raw_text=raw_text,
        ingredients=ingredients,
        ai_ingredients=ai_ingredients,
        hidden_ingredients=hidden_ingredients,
        allergens=merged_allergens,
        ai_allergens=ai_allergens,
        ingredient_breakdown=ingredient_breakdown,
    )


@app.post("/predict-risk", response_model=PredictResponse)
def predict_risk(payload: PredictRequest) -> PredictResponse:
    features = model.build_feature_vector(
        vhi=payload.vhi,
        cas=payload.cas,
        erf=payload.erf,
        ingredients=payload.ingredients,
    )
    prediction = model.predict(features, payload.ingredients)
    ai_explanation = generate_explanation(prediction, features, payload.ingredients)
    risk_reasoning = generate_risk_reasoning(prediction, features, payload.ingredients)
    recommendations = generate_recommendations(prediction, features, payload.ingredients)

    logger.info("Predict risk input: %s", payload.dict())
    prediction["ai_explanation"] = ai_explanation
    prediction["risk_reasoning"] = risk_reasoning
    prediction["recommendations"] = recommendations
    prediction["confidence_percent"] = float(prediction["probability"] * 100)
    logger.info("Predict risk output: %s", prediction)
    return PredictResponse(**prediction)


@app.get("/explain", response_model=ExplainResponse)
def explain(
    vhi: float = Query(..., ge=0.0, le=1.0),
    cas: float = Query(..., ge=0.0, le=1.0),
    erf: float = Query(..., ge=0.0, le=1.0),
    ingredients: str = Query(""),
) -> ExplainResponse:
    ingredient_list = [x.strip() for x in ingredients.split(",") if x.strip()]
    features = model.build_feature_vector(vhi=vhi, cas=cas, erf=erf, ingredients=ingredient_list)
    explanation = model.explain(features)
    return ExplainResponse(**explanation)
