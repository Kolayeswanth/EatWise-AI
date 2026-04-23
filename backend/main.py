from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.ai_service import (
    analyze_ingredients_with_ai,
    extract_ingredients_from_ocr_text,
    generate_explanation,
    generate_recommendations,
    generate_risk_reasoning,
)
from backend.food_ai_service import detect_food_names, generate_food_ingredients
from backend.db import create_user, get_user, update_user
from backend.ml_service import HybridRiskModel
from backend.ocr_service import extract_ingredients_from_image
from backend.schemas import (
    AnalyzeImageResponse,
    DetectFoodNameResponse,
    ExplainResponse,
    FoodIngredientsRequest,
    FoodIngredientsResponse,
    PredictRequest,
    PredictResponse,
    UserCreateRequest,
    UserResponse,
    UserUpdateRequest,
)

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


@app.post("/user/create", response_model=UserResponse)
def user_create(payload: UserCreateRequest) -> UserResponse:
    try:
        created = create_user(payload.dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not created or not created.get("id"):
        raise HTTPException(status_code=500, detail="Unable to create user profile")

    return UserResponse(**created)


@app.get("/user/{user_id}", response_model=UserResponse)
def user_get(user_id: str) -> UserResponse:
    try:
        user = get_user(user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(**user)


@app.put("/user/{user_id}", response_model=UserResponse)
def user_update(user_id: str, payload: UserUpdateRequest) -> UserResponse:
    try:
        updated = update_user(user_id, payload.dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not updated:
        raise HTTPException(status_code=404, detail="User not found")

    updated["id"] = user_id
    return UserResponse(**updated)


@app.post("/analyze-image", response_model=AnalyzeImageResponse)
async def analyze_image(file: UploadFile = File(...)) -> AnalyzeImageResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image file.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image file.")

    ocr_result = extract_ingredients_from_image(content)
    raw_text = str(ocr_result.get("raw_text", ""))
    lines = [str(x) for x in ocr_result.get("lines", []) if str(x).strip()]
    source = str(ocr_result.get("source", "fallback"))

    ingredients = extract_ingredients_from_ocr_text(raw_text=raw_text, lines=lines)
    if not ingredients:
        ingredients = ["ingredient detection unavailable"]
    ai_result = analyze_ingredients_with_ai(raw_text=raw_text, fallback_ingredients=ingredients)

    ai_ingredients = ai_result.get("ai_ingredients", ingredients) or ingredients
    hidden_ingredients = ai_result.get("hidden_ingredients", [])
    ai_allergens = ai_result.get("ai_allergens", [])
    ingredient_breakdown = ai_result.get("ingredient_breakdown", [])

    logger.info(
        "Analyze image input: source=%s raw_text=%s ingredients=%s ai_result=%s",
        source,
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
        lines=lines,
        source=source,
        ingredients=ingredients,
        ai_ingredients=ai_ingredients,
        hidden_ingredients=hidden_ingredients,
        allergens=merged_allergens,
        ai_allergens=ai_allergens,
        ingredient_breakdown=ingredient_breakdown,
    )


@app.post("/detect-food-name", response_model=DetectFoodNameResponse)
async def detect_food_name(file: UploadFile = File(...)) -> DetectFoodNameResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image file.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image file.")

    try:
        food_names = detect_food_names(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return DetectFoodNameResponse(food_names=food_names)


@app.post("/food-ingredients", response_model=FoodIngredientsResponse)
def food_ingredients(payload: FoodIngredientsRequest) -> FoodIngredientsResponse:
    food_name = str(payload.food_name).strip()
    language = str(payload.language or "English").strip() or "English"
    if not food_name:
        raise HTTPException(status_code=400, detail="food_name is required")

    try:
        ingredients = generate_food_ingredients(food_name=food_name, language=language)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FoodIngredientsResponse(food_name=food_name, language=language, ingredients=ingredients)


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
    prediction["risk_score"] = float(prediction["probability"])
    prediction["risk_level"] = str(prediction["risk_classification"])
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
