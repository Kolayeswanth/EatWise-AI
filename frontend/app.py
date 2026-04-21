import json
import os
from pathlib import Path
from typing import List

import pandas as pd
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "models" / "training_metrics.json"


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


TRAINING_METRICS = load_metrics()

st.set_page_config(page_title="Food Safety Risk Predictor", page_icon="F", layout="wide")

if "ingredients" not in st.session_state:
    st.session_state.ingredients = []
if "allergens" not in st.session_state:
    st.session_state.allergens = []
if "ocr_raw_text" not in st.session_state:
    st.session_state.ocr_raw_text = ""
if "ai_ingredients" not in st.session_state:
    st.session_state.ai_ingredients = []
if "hidden_ingredients" not in st.session_state:
    st.session_state.hidden_ingredients = []
if "ingredient_breakdown" not in st.session_state:
    st.session_state.ingredient_breakdown = []
if "ai_allergens" not in st.session_state:
    st.session_state.ai_allergens = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


def badge(label: str, variant: str) -> str:
    return f"<span class='badge badge-{variant}'>{label}</span>"


def render_metric_cards(items: List[tuple]) -> None:
    cols = st.columns(len(items))
    for col, (title, value, color) in zip(cols, items):
        with col:
            st.markdown(
                f"<div class='risk-card'><div class='muted'>{title}</div><div class='metric' style='color:{color}'>{value}</div></div>",
                unsafe_allow_html=True,
            )

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Fraunces:opsz,wght@9..144,700&display=swap');
    :root {
        --bg1: #f4f8f4;
        --bg2: #e8f1ed;
        --ink: #11221d;
        --card: rgba(255, 255, 255, 0.68);
        --stroke: rgba(17, 34, 29, 0.12);
        --accent: #0f766e;
        --accent-soft: #22a394;
        --good: #198754;
        --warn: #b97800;
        --danger: #bf1f4b;
    }

    .stApp {
        background:
            radial-gradient(circle at 92% 14%, rgba(15, 118, 110, 0.22), transparent 36%),
            radial-gradient(circle at 8% 88%, rgba(34, 163, 148, 0.22), transparent 40%),
            linear-gradient(165deg, var(--bg1) 0%, var(--bg2) 100%);
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(120deg, #0f766e 0%, #0c5f58 55%, #0a4a45 100%);
        color: #f8fffd;
        border-radius: 22px;
        padding: 1.35rem 1.4rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 26px 60px rgba(15, 118, 110, 0.28);
        border: 1px solid rgba(255, 255, 255, 0.18);
        animation: enterUp 520ms cubic-bezier(.2, .7, .1, 1);
    }

    .hero::after {
        content: "";
        position: absolute;
        right: -22px;
        top: -30px;
        width: 190px;
        height: 190px;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(255,255,255,0.34) 0%, rgba(255,255,255,0) 66%);
    }

    .hero-grid {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 1rem;
        align-items: center;
    }

    .hero-kpis {
        display: flex;
        gap: .7rem;
        flex-wrap: wrap;
    }

    .chip {
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.28);
        border-radius: 999px;
        padding: .3rem .7rem;
        font-size: .85rem;
        font-weight: 600;
    }

    .panel {
        border-radius: 18px;
        border: 1px solid var(--stroke);
        background: var(--card);
        backdrop-filter: blur(8px);
        box-shadow: 0 14px 30px rgba(17, 34, 29, 0.08);
        padding: 1rem;
        animation: enterUp 560ms cubic-bezier(.2, .7, .1, 1);
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.1rem;
        margin-bottom: .2rem;
        font-weight: 700;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: .8rem;
        margin-top: .65rem;
    }

    .risk-card {
        border-radius: 14px;
        border: 1px solid var(--stroke);
        background: rgba(255, 255, 255, 0.82);
        padding: .9rem;
        min-height: 112px;
        transition: transform .2s ease, box-shadow .2s ease;
    }

    .risk-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(17, 34, 29, 0.11);
    }

    .metric {
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.2;
    }

    .badge-row {
        display: flex;
        gap: .55rem;
        margin: .55rem 0 .95rem;
        flex-wrap: wrap;
    }

    .badge {
        border-radius: 999px;
        padding: .34rem .72rem;
        font-size: .82rem;
        font-weight: 700;
        border: 1px solid rgba(0,0,0,.08);
    }

    .badge-good { background: rgba(25, 135, 84, .15); color: #12613d; }
    .badge-warn { background: rgba(185, 120, 0, .14); color: #8a5a00; }
    .badge-danger { background: rgba(191, 31, 75, .14); color: #8f1637; }

    .gauge-wrap {
        display: flex;
        justify-content: center;
        margin: .35rem 0 .9rem;
    }

    .gauge {
        --p: 0;
        --g: #198754;
        width: 230px;
        height: 230px;
        border-radius: 50%;
        background: conic-gradient(var(--g) calc(var(--p) * 1%), rgba(17,34,29,.14) 0);
        display: grid;
        place-items: center;
        box-shadow: 0 10px 30px rgba(0,0,0,.14), inset 0 0 0 1px rgba(255,255,255,.2);
        animation: spinIn .75s cubic-bezier(.2,.7,.1,1);
    }

    .gauge::before {
        content: "";
        width: 170px;
        height: 170px;
        border-radius: 50%;
        background: rgba(255,255,255,.93);
        box-shadow: inset 0 0 0 1px rgba(17,34,29,.08);
    }

    .gauge-label {
        position: absolute;
        text-align: center;
        line-height: 1.15;
    }

    .gauge-score {
        font-size: 2rem;
        font-weight: 800;
    }

    .loader-wrap {
        display: flex;
        align-items: center;
        gap: .65rem;
        background: rgba(15, 118, 110, 0.1);
        border: 1px solid rgba(15, 118, 110, 0.2);
        padding: .55rem .8rem;
        border-radius: 10px;
        margin-top: .45rem;
        margin-bottom: .45rem;
        font-size: .92rem;
        color: #0c5f58;
        font-weight: 600;
    }

    .loader {
        width: 17px;
        height: 17px;
        border-radius: 50%;
        border: 3px solid rgba(15,118,110,.2);
        border-top-color: #0f766e;
        animation: rotate 1s linear infinite;
    }

    .meter-shell {
        margin-top: .6rem;
        height: 12px;
        border-radius: 999px;
        background: linear-gradient(90deg, #198754 0%, #f0b400 50%, #bf1f4b 100%);
        position: relative;
        overflow: hidden;
    }

    .meter-dot {
        position: absolute;
        top: 50%;
        width: 18px;
        height: 18px;
        border-radius: 999px;
        border: 2px solid #ffffff;
        transform: translate(-50%, -50%);
        box-shadow: 0 2px 9px rgba(0,0,0,0.22);
        animation: pulse 1.8s ease-in-out infinite;
    }

    .muted {
        color: rgba(17, 34, 29, 0.72);
        font-size: .93rem;
    }

    @keyframes enterUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulse {
        0%, 100% { transform: translate(-50%, -50%) scale(1); }
        50% { transform: translate(-50%, -50%) scale(1.09); }
    }

    @keyframes rotate {
        to { transform: rotate(360deg); }
    }

    @keyframes spinIn {
        from { opacity: 0; transform: scale(.9) rotate(-18deg); }
        to { opacity: 1; transform: scale(1) rotate(0deg); }
    }

    @media (max-width: 980px) {
        .hero-grid {
            grid-template-columns: 1fr;
        }
        .metric-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-grid">
        <div>
          <h1 style="font-family: 'Fraunces', serif; margin: 0 0 .35rem 0; line-height: 1.12; font-size: 2.15rem;">FoodShield AI 🛡️</h1>
          <p style="margin: 0; font-size: 1.05rem; max-width: 780px;">A demo-grade food safety assistant that scans labels, explains ingredients, predicts risk, and maps the result to our research paper.</p>
        </div>
        <div class="hero-kpis">
          <span class="chip">📷 OCR</span>
          <span class="chip">🤖 Gemini AI</span>
          <span class="chip">🧠 Hybrid ML</span>
          <span class="chip">🔍 SHAP</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

scan_tab, results_tab, ai_tab, about_tab = st.tabs(["Scan", "Results", "AI Insights", "About"])

with scan_tab:
    left, right = st.columns([1.15, 0.95], gap="large")

    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📦 Scan Label</div>", unsafe_allow_html=True)
        st.caption("Upload a food package image to extract ingredients.")
        uploaded = st.file_uploader("Drop image or browse", type=["png", "jpg", "jpeg", "webp"])

        if uploaded is not None:
            st.image(uploaded, caption="Uploaded label", use_container_width=True)
            extract_clicked = st.button("🔎 Extract Ingredients", use_container_width=True)
            if extract_clicked:
                loading_box = st.empty()
                loading_box.markdown("<div class='loader-wrap'><div class='loader'></div>Reading label with OCR...</div>", unsafe_allow_html=True)
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                response = requests.post(f"{API_BASE}/analyze-image", files=files, timeout=180)
                loading_box.empty()

                if response.ok:
                    payload = response.json()
                    st.session_state.ingredients = payload.get("ingredients", [])
                    st.session_state.allergens = payload.get("allergens", [])
                    st.session_state.ocr_raw_text = payload.get("raw_text", "")
                    st.session_state.ai_ingredients = payload.get("ai_ingredients", [])
                    st.session_state.hidden_ingredients = payload.get("hidden_ingredients", [])
                    st.session_state.ingredient_breakdown = payload.get("ingredient_breakdown", [])
                    st.session_state.ai_allergens = payload.get("ai_allergens", [])
                    st.success("✅ Extraction completed")
                else:
                    st.error(f"OCR failed: {response.text}")

        ingredients: List[str] = st.session_state.ingredients
        allergens: List[str] = st.session_state.allergens

        if ingredients:
            st.markdown("### 🧾 Extracted Ingredients")
            st.markdown(f"<p class='muted'>{', '.join(ingredients)}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='muted'>No ingredients detected yet. Upload an image and extract ingredients.</p>", unsafe_allow_html=True)

        if st.session_state.ai_ingredients:
            st.markdown("### 🤖 Smart Ingredient Breakdown")
            st.markdown(f"<p class='muted'>{', '.join(st.session_state.ai_ingredients)}</p>", unsafe_allow_html=True)

        if st.session_state.hidden_ingredients:
            st.info(f"🕵️ Hidden ingredients detected: {', '.join(st.session_state.hidden_ingredients)}")

        if st.session_state.ai_allergens:
            st.warning(f"🧠 AI allergen check: {', '.join(st.session_state.ai_allergens)}")

        if allergens:
            st.error(f"⚠️ Allergen alert: {', '.join(allergens)}")
        elif ingredients:
            st.info("✅ No major allergens detected in extracted ingredients")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>⚙️ Risk Context</div>", unsafe_allow_html=True)
        st.caption("Tune the context inputs from field conditions.")
        vhi = st.slider("Vendor Hygiene Index (VHI)", 0.0, 1.0, 0.65, 0.01)
        cas = st.slider("Consumer Awareness Score (CAS)", 0.0, 1.0, 0.55, 0.01)
        erf = st.slider("Environmental Risk Factor (ERF)", 0.0, 1.0, 0.5, 0.01)

        predict_clicked = st.button("🚀 Predict Safety Risk", use_container_width=True)
        if predict_clicked:
            loading_box = st.empty()
            loading_box.markdown("<div class='loader-wrap'><div class='loader'></div>Running hybrid model prediction...</div>", unsafe_allow_html=True)
            payload = {
                "ingredients": st.session_state.ingredients,
                "vhi": vhi,
                "cas": cas,
                "erf": erf,
            }
            response = requests.post(f"{API_BASE}/predict-risk", json=payload, timeout=60)
            loading_box.empty()

            if not response.ok:
                st.error(f"Prediction failed: {response.text}")
            else:
                st.session_state.last_prediction = response.json()

        if st.session_state.last_prediction:
            result = st.session_state.last_prediction
            cls = result["risk_classification"]
            prob = float(result["probability"])
            color = "#2a6f3f" if cls == "Low" else "#d38b00" if cls == "Medium" else "#9f1d35"
            meter_label = "SAFE" if cls == "Low" else "MODERATE" if cls == "Medium" else "HIGH RISK"
            meter_icon = "🟢" if cls == "Low" else "🟡" if cls == "Medium" else "🔴"

            st.markdown("<div class='badge-row'>" + badge("🟢 Safe", "good") + badge("🟡 Moderate", "warn") + badge("🔴 High Risk", "danger") + "</div>", unsafe_allow_html=True)
            st.markdown("<div class='gauge-wrap'>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style='position:relative;'>
                    <div class='gauge' style='--p:{prob * 100:.2f}; --g:{color};'></div>
                    <div class='gauge-label'>
                        <div class='muted'>Risk Score</div>
                        <div class='gauge-score' style='color:{color};'>{prob * 100:.0f}%</div>
                        <div style='font-weight:700; color:{color};'>{meter_icon} {meter_label}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            render_metric_cards([
                ("Risk Class", cls, color),
                ("Confidence", f"{prob:.1%}", color),
                ("Safety Status", meter_label, color),
            ])
            st.markdown(
                f"<div class='meter-shell'><span class='meter-dot' style='left:{max(2, min(98, prob * 100)):.2f}%; background:{color};'></span></div>",
                unsafe_allow_html=True,
            )
            st.caption(f"💡 {result.get('explanation', '')}")

        st.markdown("</div>", unsafe_allow_html=True)

with results_tab:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Results</div>", unsafe_allow_html=True)
    st.caption("Paper reference values are shown alongside the trained demo model metrics.")

    demo_accuracy = float(TRAINING_METRICS.get("classification_report", {}).get("accuracy", 0.776))
    demo_f1 = float(TRAINING_METRICS.get("classification_report", {}).get("weighted avg", {}).get("f1-score", 0.7721883674933927))
    demo_auc = float(TRAINING_METRICS.get("roc_auc_ovr", 0.9189395046135461))
    paper_accuracy = 0.946

    render_metric_cards([
        ("Research paper accuracy", f"{paper_accuracy:.1%}", "#0f766e"),
        ("Demo weighted F1", f"{demo_f1:.1%}", "#2a6f3f"),
        ("Demo AUC", f"{demo_auc:.1%}", "#b97800"),
    ])

    comparison = pd.DataFrame(
        {
            "Metric": ["Paper Accuracy", "Demo Accuracy", "Demo F1", "Demo AUC"],
            "Value": [paper_accuracy * 100, demo_accuracy * 100, demo_f1 * 100, demo_auc * 100],
        }
    ).set_index("Metric")
    st.bar_chart(comparison)

    if st.session_state.last_prediction:
        st.markdown("### Latest prediction")
        latest = st.session_state.last_prediction
        st.write(f"Class: {latest.get('risk_classification', 'N/A')}")
        st.write(f"Confidence: {latest.get('confidence_percent', 0):.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

with ai_tab:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🤖 AI Insights</div>", unsafe_allow_html=True)
    if st.session_state.last_prediction:
        result = st.session_state.last_prediction
        st.markdown(f"<div class='risk-card'><div class='muted'>Simple explanation</div><div style='font-size:1rem; font-weight:600;'>{result.get('ai_explanation', result.get('explanation', ''))}</div></div>", unsafe_allow_html=True)

        st.markdown("### ✅ Recommendations")
        recommendations = result.get("recommendations", [])
        for item in recommendations or ["Store food below 5°C when possible.", "Prefer products with fewer preservatives.", "Check allergen labels carefully."]:
            st.markdown(f"- {item}")

        st.markdown("### 🧾 Smart Ingredient Breakdown")
        breakdown = st.session_state.ingredient_breakdown
        if breakdown:
            st.dataframe(pd.DataFrame(breakdown), use_container_width=True)
        else:
            st.info("Run OCR first to see the smart ingredient breakdown.")
    else:
        st.info("Run a prediction first to generate AI insights and recommendations.")
    st.markdown("</div>", unsafe_allow_html=True)

with about_tab:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>ℹ️ About This Project</div>", unsafe_allow_html=True)
    st.markdown("**Hybrid Ensemble Machine Learning Framework for Predictive Consumer Food Safety Risk Assessment**")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("### Problem Statement")
        st.markdown("- Food safety risks are hard to spot from labels alone.")
        st.markdown("- Consumers often do not know how ingredients and storage conditions affect risk.")
        st.markdown("- Many apps stop at OCR and do not explain what the result means.")

    with c2:
        st.markdown("### What This App Does")
        st.markdown("- 📷 OCR → Extracts ingredients from a food label image")
        st.markdown("- 🤖 AI → Normalizes ingredients and detects hidden additives/allergens")
        st.markdown("- 🧠 ML → Predicts contamination risk with a hybrid ensemble")
        st.markdown("- 🔍 SHAP → Explains which features influenced the prediction")

    st.divider()
    st.markdown("### How This Matches Our Research Paper")
    paper_map = pd.DataFrame(
        {
            "Paper idea": [
                "Hybrid model",
                "Features",
                "Dataset",
                "Evaluation",
                "Explainability",
            ],
            "Implementation": [
                "Random Forest + Gradient Boosting + SVM",
                "VHI, CAS, ERF",
                "Synthetic dataset aligned to the research design",
                "Accuracy, F1, and AUC are tracked and displayed",
                "SHAP is used for feature importance and explanation",
            ],
        }
    )
    st.dataframe(paper_map, use_container_width=True, hide_index=True)

    st.markdown("### Paper-Linked Results")
    st.markdown("- Accuracy: ~94.6% from the research paper reference")
    st.markdown(f"- Demo weighted F1: {float(TRAINING_METRICS.get('classification_report', {}).get('weighted avg', {}).get('f1-score', 0.7721883674933927)):.1%}")
    st.markdown(f"- Demo AUC: {float(TRAINING_METRICS.get('roc_auc_ovr', 0.9189395046135461)):.1%}")
    st.markdown("- SHAP: used for feature importance and simple explanation")

    st.divider()
    st.markdown("### Deployment Notes")
    st.markdown("- Backend can be deployed on Render using the provided `Procfile` or `render.yaml`.")
    st.markdown("- Frontend can be deployed on Streamlit Cloud by setting `API_BASE` to the deployed backend URL.")
    st.markdown("- Secrets should be stored as environment variables: `GEMINI_API_KEY`, `API_BASE`.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Backend API: configurable with API_BASE | Gemini AI enabled | Demo mode ready")
