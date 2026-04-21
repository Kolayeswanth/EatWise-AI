import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

FEATURE_COLUMNS = [
    "vhi",
    "cas",
    "erf",
    "ingredient_risk",
    "allergen_count",
    "ingredient_count",
]
TARGET_COLUMN = "target"


RISK_KEYWORDS = {
    "raw": 0.8,
    "unpasteurized": 0.9,
    "nitrate": 0.55,
    "benzoate": 0.5,
    "sulfite": 0.65,
    "color": 0.35,
    "flavor": 0.3,
    "preservative": 0.45,
    "hydrogenated": 0.75,
    "syrup": 0.4,
    "oil": 0.25,
    "salt": 0.2,
    "sugar": 0.2,
    "milk": 0.2,
    "egg": 0.2,
}

ALLERGENS = [
    "peanut",
    "nut",
    "almond",
    "walnut",
    "gluten",
    "wheat",
    "soy",
    "milk",
    "egg",
    "fish",
    "shellfish",
    "sesame",
]


def ingredient_risk_score(ingredient_string: str) -> float:
    tokens = ingredient_string.lower().replace("(", " ").replace(")", " ").replace(",", " ").split()
    if not tokens:
        return 0.0
    scores = [RISK_KEYWORDS.get(token, 0.05) for token in tokens]
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def count_allergens(ingredient_string: str) -> int:
    text = ingredient_string.lower()
    return sum(1 for allergen in ALLERGENS if allergen in text)


def build_synthetic_dataset(n_samples: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    ingredient_templates = [
        "wheat flour, sugar, salt, soy lecithin, flavor",
        "raw milk, sugar syrup, preservative benzoate, color",
        "rice flour, palm oil, salt, spice",
        "unpasteurized cream, egg powder, sulfite, artificial flavor",
        "corn flour, peanut oil, whey milk powder, salt",
        "fish extract, sesame oil, nitrate preservative",
        "almond paste, sugar, cocoa butter, milk solids",
        "wheat gluten, soybean oil, stabilizer, emulsifier",
    ]

    rows = []
    for _ in range(n_samples):
        ingredient_string = ingredient_templates[rng.integers(0, len(ingredient_templates))]

        vhi = float(np.clip(rng.normal(0.6, 0.18), 0.0, 1.0))
        cas = float(np.clip(rng.normal(0.55, 0.2), 0.0, 1.0))
        erf = float(np.clip(rng.normal(0.5, 0.22), 0.0, 1.0))

        i_risk = ingredient_risk_score(ingredient_string)
        allergen_count = count_allergens(ingredient_string)
        ingredient_count = max(3, len([x for x in ingredient_string.split(",") if x.strip()]))

        raw_score = (
            1.15 * i_risk
            + 0.95 * erf
            + 0.6 * (1 - vhi)
            + 0.45 * (1 - cas)
            + 0.1 * allergen_count
            + rng.normal(0, 0.12)
        )
        probability = 1.0 / (1.0 + np.exp(-4.2 * (raw_score - 1.1)))

        if probability < 0.33:
            target = 0
        elif probability < 0.66:
            target = 1
        else:
            target = 2

        rows.append(
            {
                "vhi": vhi,
                "cas": cas,
                "erf": erf,
                "ingredient_risk": i_risk,
                "allergen_count": allergen_count,
                "ingredient_count": ingredient_count,
                "ingredients": ingredient_string,
                "target": target,
            }
        )

    return pd.DataFrame(rows)


def train_and_save() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = build_synthetic_dataset()
    csv_path = DATA_DIR / "synthetic_food_safety_dataset.csv"
    df.to_csv(csv_path, index=False)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    rf = RandomForestClassifier(n_estimators=220, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    svm = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(probability=True, kernel="rbf", C=2.0, gamma="scale", random_state=42)),
        ]
    )

    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    weights = np.array([0.4, 0.35, 0.25], dtype=float)

    rf_proba = rf.predict_proba(X_test)
    gb_proba = gb.predict_proba(X_test)
    svm_proba = svm.predict_proba(X_test)

    blended = weights[0] * rf_proba + weights[1] * gb_proba + weights[2] * svm_proba
    blended = 1.0 / (1.0 + np.exp(-5 * (blended - 0.5)))
    blended = blended / blended.sum(axis=1, keepdims=True)

    y_pred = np.argmax(blended, axis=1)

    model_bundle = {
        "rf": rf,
        "gb": gb,
        "svm": svm,
        "weights": weights,
        "feature_columns": FEATURE_COLUMNS,
        "class_names": ["Low", "Medium", "High"],
        "risk_keywords": RISK_KEYWORDS,
        "allergens": ALLERGENS,
        "target_map": {0: "Low", 1: "Medium", 2: "High"},
    }

    bundle_path = MODELS_DIR / "hybrid_ensemble.joblib"
    joblib.dump(model_bundle, bundle_path)

    metrics = {
        "dataset_path": str(csv_path),
        "bundle_path": str(bundle_path),
        "roc_auc_ovr": float(roc_auc_score(y_test, blended, multi_class="ovr")),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    (MODELS_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training complete")
    print(json.dumps({"bundle": str(bundle_path), "auc": metrics["roc_auc_ovr"]}, indent=2))


if __name__ == "__main__":
    train_and_save()
