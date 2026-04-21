from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "models" / "hybrid_ensemble.joblib"


class HybridRiskModel:
    def __init__(self) -> None:
        if not BUNDLE_PATH.exists():
            raise FileNotFoundError(f"Model bundle not found: {BUNDLE_PATH}")
        bundle = joblib.load(BUNDLE_PATH)
        self.rf = bundle["rf"]
        self.gb = bundle["gb"]
        self.svm = bundle["svm"]
        self.weights = np.array(bundle["weights"], dtype=float)
        self.feature_columns = bundle["feature_columns"]
        self.class_names = bundle["class_names"]
        self.risk_keywords = bundle["risk_keywords"]
        self.allergens = bundle["allergens"]

        self._rf_explainer = shap.TreeExplainer(self.rf)

    def ingredient_risk_score(self, ingredients: List[str]) -> float:
        if not ingredients:
            return 0.05

        tokens = " ".join(ingredients).lower().replace("(", " ").replace(")", " ").replace(",", " ").split()
        scores = [self.risk_keywords.get(token, 0.05) for token in tokens] or [0.05]
        return float(np.clip(np.mean(scores), 0.0, 1.0))

    def detect_allergens(self, ingredients: List[str]) -> List[str]:
        text = " ".join(ingredients).lower()
        return [a for a in self.allergens if a in text]

    def build_feature_vector(self, vhi: float, cas: float, erf: float, ingredients: List[str]) -> Dict[str, float]:
        detected_allergens = self.detect_allergens(ingredients)
        ingredient_count = max(1, len([item for item in ingredients if item.strip()]))

        return {
            "vhi": float(vhi),
            "cas": float(cas),
            "erf": float(erf),
            "ingredient_risk": self.ingredient_risk_score(ingredients),
            "allergen_count": float(len(detected_allergens)),
            "ingredient_count": float(ingredient_count),
        }

    def _ensemble_probabilities(self, X_df: pd.DataFrame) -> np.ndarray:
        rf_p = self.rf.predict_proba(X_df)
        gb_p = self.gb.predict_proba(X_df)
        svm_p = self.svm.predict_proba(X_df)

        blended = self.weights[0] * rf_p + self.weights[1] * gb_p + self.weights[2] * svm_p
        blended = 1.0 / (1.0 + np.exp(-5 * (blended - 0.5)))
        blended = blended / blended.sum(axis=1, keepdims=True)
        return blended

    def predict(self, features: Dict[str, float], ingredients: List[str]) -> Dict:
        X_df = pd.DataFrame([features], columns=self.feature_columns)
        blended = self._ensemble_probabilities(X_df)[0]

        class_index = int(np.argmax(blended))
        class_name = self.class_names[class_index]
        probability = float(blended[class_index])

        explanation = (
            f"Risk is {class_name} mainly due to ingredient risk {features['ingredient_risk']:.2f}, "
            f"ERF {features['erf']:.2f}, and allergen count {int(features['allergen_count'])}."
        )

        return {
            "probability": probability,
            "risk_classification": class_name,
            "class_probabilities": {
                self.class_names[i]: float(blended[i]) for i in range(len(self.class_names))
            },
            "features": features,
            "allergens_detected": self.detect_allergens(ingredients),
            "explanation": explanation,
        }

    def explain(self, features: Dict[str, float]) -> Dict:
        X_df = pd.DataFrame([features], columns=self.feature_columns)
        shap_values = self._rf_explainer.shap_values(X_df)

        if isinstance(shap_values, list):
            stacked = np.stack(shap_values, axis=0)
            values = np.mean(np.abs(stacked), axis=(0, 1))
        else:
            arr = np.array(shap_values)
            if arr.ndim == 1:
                values = np.abs(arr)
            elif arr.ndim == 2:
                values = np.mean(np.abs(arr), axis=0)
            elif arr.ndim == 3:
                # Aggregate over non-feature axes to get one importance per feature.
                feature_axis = next((i for i, s in enumerate(arr.shape) if s == len(self.feature_columns)), -1)
                if feature_axis == -1:
                    values = np.mean(np.abs(arr), axis=tuple(range(arr.ndim - 1)))
                else:
                    axes = tuple(i for i in range(arr.ndim) if i != feature_axis)
                    values = np.mean(np.abs(arr), axis=axes)
            else:
                values = np.mean(np.abs(arr.reshape(-1, len(self.feature_columns))), axis=0)

        values = np.array(values).reshape(-1)

        ranking = sorted(
            [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in zip(self.feature_columns, values)
            ],
            key=lambda x: x["importance"],
            reverse=True,
        )

        top = ", ".join(item["feature"] for item in ranking[:3])
        base_value = self._rf_explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(np.mean(base_value))

        return {
            "feature_importance": ranking,
            "summary": f"Top risk drivers: {top}",
            "shap_base_value": float(base_value),
        }
