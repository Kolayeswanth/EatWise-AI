# Food Safety Risk Prediction App

Hybrid AI demo for food label safety analysis.

## Overview

- Scan a food label image
- Extract ingredients with OCR
- Normalize ingredients with Gemini AI
- Predict contamination risk with a hybrid ensemble model
- Explain the prediction with SHAP + Gemini
- Show everything in a tabbed Streamlit dashboard

## Frontend tabs

- Scan
- Results
- AI Insights
- About

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
- EasyOCR
- Scikit-learn
- SHAP
- Gemini AI

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
- Add `GEMINI_API_KEY` as an environment variable.

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
- Gemini AI integration is active.
- UI is tabbed, demo-ready, and aligned with the paper.
