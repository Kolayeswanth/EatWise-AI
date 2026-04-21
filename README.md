# Food Safety Risk Prediction App

Hybrid AI assistant for food label safety analysis.

## Overview

- Scan a food label image
- Extract ingredients with Azure Vision Read OCR (primary)
- Normalize ingredients with deterministic rule-based logic
- Predict contamination risk with a hybrid ensemble model
- Explain the prediction with SHAP and optional Gemini narration
- Guide users through a step-by-step assistant-style Streamlit experience

## Frontend guided flow

- Login and onboarding
- Profile setup (health conditions and allergies)
- Scan (camera or upload)
- Ingredient confirmation and edit
- Translation to user preferred language
- Risk analysis and personalized result cards

## Personalization

- User profile includes name, age, health conditions, and allergies.
- Profile is stored in Streamlit session state for the current run.
- Personalized alert is shown when detected allergens match user allergies.

## Research paper mapping

- Hybrid model: Random Forest + Gradient Boosting + SVM
- Features: Vendor Hygiene Index (VHI), Consumer Awareness Score (CAS), Environmental Risk Factor (ERF)
- Dataset: synthetic dataset aligned with the research design
- Evaluation: accuracy, F1, and AUC are tracked in the demo
- Explainability: SHAP feature importance is included

## Tech stack

- FastAPI
- Streamlit
- Azure Vision Read API
- Scikit-learn
- SHAP
- Gemini AI (optional, explanation only)

## Deployment note

- OCR is Azure-primary with bounded polling and a lightweight local text fallback.
- Gemini is optional for explanations only, never required for ingredient parsing or endpoint success.

## Results shown in the app

- Research paper accuracy: ~94.6%
- Demo weighted F1: ~77.2%
- Demo AUC: ~91.9%

## Project structure

- backend/ - API, OCR, ML, and AI services
- frontend/ - Streamlit app
- scripts/ - training and validation scripts
- models/ - trained model and metrics
- data/ - dataset and sample image files

## Key files

- backend/main.py
- backend/ai_service.py
- backend/ml_service.py
- backend/ocr_service.py
- backend/schemas.py
- frontend/app.py
- scripts/train_model.py
- Procfile
- render.yaml

## Environment variables

- `GEMINI_API_KEY`
- `GEMINI_MODEL` (optional, default: `gemini-2.5-flash`)
- `AZURE_VISION_ENDPOINT`
- `AZURE_VISION_KEY`
- `USE_GEMINI` (optional, default: `true`)
- `API_BASE` for the frontend backend URL (also supported via Streamlit secrets)

## Run locally

Backend:

```powershell
c:/Users/Kola_Yeswanth/OneDrive/Desktop/fyp/.venv/Scripts/python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
c:/Users/Kola_Yeswanth/OneDrive/Desktop/fyp/.venv/Scripts/python.exe -m streamlit run frontend/app.py --server.port 8501 --server.address 127.0.0.1
```

## Deployment

Backend on Render:

- Connect the repo on Render.
- Use the provided `render.yaml` or the start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
- Add `AZURE_VISION_ENDPOINT` and `AZURE_VISION_KEY` environment variables.
- Add `GEMINI_API_KEY` only if Gemini enhancements are desired.

Frontend on Streamlit Cloud:

- Deploy `frontend/app.py`.
- Set `API_BASE` to the deployed backend URL in Streamlit secrets.
- Add `GEMINI_API_KEY` as a secret if your deployment needs direct Gemini calls.

Mobile-friendly usage:

- Turn on **Mobile-friendly layout** from the app sidebar.
- Use browser menu -> **Add to Home Screen / Install app** for app-like access.

## Current status

- Local backend is running.
- Local frontend is running.
- OCR pipeline is Azure-primary with fallback that still returns structured output.
- Ingredient parsing is deterministic, cleaned, and deduplicated; Gemini is explanation-only.
- UI is guided, assistant-like, and aligned with the paper.
