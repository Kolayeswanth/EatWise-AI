from typing import List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    ingredients: List[str] = Field(default_factory=list)
    vhi: float = Field(..., ge=0.0, le=1.0, description="Vendor Hygiene Index")
    cas: float = Field(..., ge=0.0, le=1.0, description="Consumer Awareness Score")
    erf: float = Field(..., ge=0.0, le=1.0, description="Environmental Risk Factor")


class AnalyzeImageResponse(BaseModel):
    raw_text: str
    ingredients: List[str]
    ai_ingredients: List[str] = Field(default_factory=list)
    hidden_ingredients: List[str] = Field(default_factory=list)
    allergens: List[str]
    ai_allergens: List[str] = Field(default_factory=list)
    ingredient_breakdown: List[dict] = Field(default_factory=list)


class PredictResponse(BaseModel):
    probability: float
    risk_classification: str
    class_probabilities: dict
    features: dict
    allergens_detected: List[str]
    explanation: str
    ai_explanation: str = ""
    recommendations: List[str] = Field(default_factory=list)
    confidence_percent: float = 0.0


class ExplainResponse(BaseModel):
    feature_importance: List[dict]
    summary: str
    shap_base_value: Optional[float] = None
