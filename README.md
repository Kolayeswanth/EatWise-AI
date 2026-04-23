# Food Safety Risk Prediction App

Hybrid food-label analysis app with a FastAPI backend, Streamlit frontend, and a trained ensemble risk model.

## What this project does exactly

1. Accepts a food-label image.
2. Tries OCR with Azure Vision Read API.
3. If Azure OCR fails, returns a safe structured fallback by extracting text-like lines from image bytes.
4. Extracts the ingredient section using deterministic rules (start markers, stop markers, cleaning, dedupe).
5. Normalizes ingredient tokens with rule-based mappings.
6. Detects allergens from ingredient text.
7. Builds a feature vector from user risk inputs and ingredient-derived signals.
8. Predicts risk with a weighted hybrid ensemble (Random Forest + Gradient Boosting + SVM).
9. Returns risk score, level, class probabilities, explanation, reasoning, and recommendations.
10. Shows a guided assistant-style UI in Streamlit, including profile, translation, and personalized allergy alerts.

## System architecture

- Backend: FastAPI service with endpoints for OCR analysis, prediction, explanation, and user profile CRUD.
- OCR layer: Azure Vision Read (primary) with bounded polling and timeout.
- Ingredient parser: deterministic, rule-based parser from OCR output.
- ML layer: pre-trained ensemble loaded from `models/hybrid_ensemble.joblib`.
- Explainability: SHAP (TreeExplainer on the Random Forest component).
- Optional AI text generation: Gemini for user-facing explanation/reasoning/recommendations only.
- Frontend: Streamlit multi-step flow with profile persistence in session state and backend user profile support.

## Backend behavior

### 1) OCR and ingredient extraction (`POST /analyze-image`)

Input:
- multipart form with `file` (must be an image MIME type)

Processing:
- Azure OCR request: `vision/v3.2/read/analyze`
- Polling: up to 5 attempts, 1 second interval, 5 second HTTP timeout per call
- Fallback if Azure fails: lightweight local text-like line extraction (not full OCR)
- Ingredient parsing:
	- Start markers: `ingredients`, `composition`, `склад`, `состав`
	- Stop markers: `may contain`, `contains`, `nutrition`, `nutritional information`, `allergen`, `warning`
	- Cleans numbers/symbols, filters noise tokens, normalizes variants, deduplicates
	- Caps output to 30 ingredients

Output fields:
- `raw_text`, `lines`, `source`
- `ingredients` (deterministic parsed list)
- `ai_ingredients` (rule-based normalized list)
- `hidden_ingredients`, `allergens`, `ai_allergens`, `ingredient_breakdown`

Notes:
- `analyze_ingredients_with_ai` currently runs rule-based normalization. Gemini is not required for this endpoint to succeed.

### 2) Risk prediction (`POST /predict-risk`)

Input JSON:
- `ingredients: string[]`
- `vhi: float` in [0,1]
- `cas: float` in [0,1]
- `erf: float` in [0,1]

Feature vector used by model:
- `vhi`
- `cas`
- `erf`
- `ingredient_risk` (derived from keyword risk dictionary)
- `allergen_count` (detected from ingredient text)
- `ingredient_count` (non-empty ingredient entries)

Model output:
- Ensemble probabilities from RF + GB + SVM (weighted blend, then normalized)
- Predicted class and confidence
- Backward-compatible + standardized response fields:
	- `probability`, `risk_classification`, `class_probabilities`, `features`, `allergens_detected`, `explanation`
	- `risk_score`, `risk_level`, `ai_explanation`, `risk_reasoning`, `recommendations`, `confidence_percent`

### 3) Explainability endpoint (`GET /explain`)

Inputs:
- query params: `vhi`, `cas`, `erf`, `ingredients` (comma-separated)

Returns:
- `feature_importance` (SHAP-based ranking)
- `summary` (top drivers)
- `shap_base_value`

Important detail:
- SHAP is computed with a Random Forest TreeExplainer from the hybrid bundle (not a global explainer for all three models).

### 4) User profile endpoints

- `POST /user/create`
- `GET /user/{user_id}`
- `PUT /user/{user_id}`

Profile schema includes:
- `name`, `age`, `preferred_language`, `health_conditions`, `allergies`

## Frontend behavior (Streamlit)

Guided steps:
1. Profile
2. Scan
3. Ingredients
4. Clean & Translate
5. Analyze
6. Results

What the app does in UI:
- Captures user profile and optional backend profile ID.
- Accepts uploaded/camera image, calls `/analyze-image`.
- Lets user review/edit ingredient list.
- Cleans ingredient list for UI consistency.
- Optionally translates ingredient list with LibreTranslate endpoints.
- Maps qualitative answers to numeric indices:
	- Good: 0.20
	- Average: 0.50
	- Poor: 0.85
	- Not sure: 0.55
- Calls `/predict-risk` and displays results.
- Creates personalized allergy alert when user allergies overlap with detected allergens or ingredient text.

## Gemini usage in this project

- Controlled by `USE_GEMINI` (default `true`).
- Used only for:
	- `ai_explanation`
	- `risk_reasoning`
	- `recommendations`
- If Gemini is disabled/unavailable, deterministic fallback text is returned.
- Ingredient extraction and normalization do not depend on Gemini.

## Tech stack

- FastAPI
- Streamlit
- Scikit-learn
- SHAP
- Azure Vision Read API
- Gemini API (optional)
- LibreTranslate (frontend translation fallback network services)

## Repository structure

- `backend/`: API, OCR, ML, AI text helpers, schemas
- `frontend/`: Streamlit guided app
- `models/`: trained ensemble bundle + training metrics
- `scripts/`: model training and validation helpers
- `data/`: synthetic dataset
- `mobile-android/`: Android client project

## Environment variables

Required for Azure OCR path:
- `AZURE_VISION_ENDPOINT`
- `AZURE_VISION_KEY`

Optional:
- `GEMINI_API_KEY`
- `GEMINI_MODEL` (default: `gemini-2.5-flash`)
- `USE_GEMINI` (`true`/`false`, default: `true`)
- `API_BASE` (frontend backend URL; can also come from Streamlit secrets)

## Run locally (Windows example)

Use your configured virtual environment Python executable.

Backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501 --server.address 127.0.0.1
```

## Deployment

Backend (Render):
- Use `render.yaml` or start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Configure Azure env vars for OCR
- Add Gemini key only if AI text enhancements are desired

Frontend (Streamlit Cloud):
- Deploy `frontend/app.py`
- Set `API_BASE` to backend URL in secrets/environment

## Model and demo metrics

Training and demo metrics are tracked in:
- `models/training_metrics.json`

The app currently highlights paper/demo figures in UI content, while API predictions are produced from the bundled ensemble model.
