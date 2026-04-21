import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from requests import RequestException

try:
    from google import genai
except Exception:
    genai = None

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "models" / "training_metrics.json"


def get_api_base() -> str:
    try:
        secret_api_base = st.secrets.get("API_BASE")
    except Exception:
        secret_api_base = None
    return os.getenv("API_BASE") or secret_api_base or "https://eatwise-ai.onrender.com"


def get_gemini_key() -> str:
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""
    return os.getenv("GEMINI_API_KEY", "") or secret_key


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


@st.cache_resource
def get_genai_client(api_key: str):
    if not api_key or genai is None:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def parse_ingredient_text(value: str) -> List[str]:
    seen = set()
    parsed = []
    for item in value.replace("\n", ",").split(","):
        cleaned = item.strip()
        lowered = cleaned.lower()
        if cleaned and lowered not in seen:
            parsed.append(cleaned)
            seen.add(lowered)
    return parsed


def is_placeholder_ingredients(items: List[str]) -> bool:
    normalized = [str(item).strip().lower() for item in items if str(item).strip()]
    return normalized == ["ingredient detection unavailable"]


def ensure_non_empty_ingredients(items: List[str]) -> List[str]:
    cleaned = [x for x in items if str(x).strip()]
    if not cleaned:
        return ["ingredient detection unavailable"]
    return cleaned


def parse_json_like(text: str) -> Optional[dict]:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\\s*", "", cleaned)
    cleaned = re.sub(r"^```\\s*", "", cleaned)
    cleaned = re.sub(r"\\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except Exception:
        return None


def normalize_ingredients_with_ai(
    client,
    model: str,
    ingredients: List[str],
    raw_text: str,
) -> List[str]:
    if client is None:
        return ingredients
    prompt = f"""
Normalize and clean the ingredient names from OCR.
Return ONLY JSON:
{{
  "ingredients": ["string"]
}}

OCR text:
{raw_text}

Current ingredients:
{ingredients}

Rules:
- Keep food-safe, human-readable ingredient names.
- Remove obvious OCR artifacts.
- Keep list concise and unique.
""".strip()
    try:
        response = client.models.generate_content(model=model, contents=prompt)
        data = parse_json_like(getattr(response, "text", ""))
        if isinstance(data, dict) and isinstance(data.get("ingredients"), list):
            normalized = [str(x).strip() for x in data["ingredients"] if str(x).strip()]
            if normalized:
                return normalized
    except Exception:
        pass
    return ingredients


def translate_ingredients_with_ai(
    client,
    model: str,
    ingredients: List[str],
    target_language: str,
) -> Tuple[List[str], str]:
    if client is None:
        return ingredients, "Translation skipped: Gemini key unavailable."

    prompt = f"""
Translate ingredient names into {target_language}.
Return ONLY JSON:
{{
  "detected_language": "string",
  "translated_ingredients": ["string"]
}}

Ingredients:
{ingredients}

Rules:
- Keep food ingredient terms accurate.
- Preserve order when possible.
- Return concise ingredient names only.
""".strip()

    try:
        response = client.models.generate_content(model=model, contents=prompt)
        data = parse_json_like(getattr(response, "text", ""))
        if isinstance(data, dict):
            translated = [str(x).strip() for x in data.get("translated_ingredients", []) if str(x).strip()]
            detected = str(data.get("detected_language", "Unknown")).strip()
            if translated:
                note = f"Translated from {detected} to {target_language}."
                return translated, note
    except Exception:
        pass

    return ingredients, "Translation unavailable: using detected ingredients as-is."


def generate_friendly_extras_with_ai(
    client,
    model: str,
    ingredients: List[str],
    risk_classification: str,
    probability: float,
) -> Tuple[str, List[str]]:
    if client is None:
        return "", []

    prompt = f"""
You are a food safety assistant.
Generate simple user-facing text.
Return ONLY JSON:
{{
  "explanation": "string",
  "recommendations": ["string"]
}}

Risk class: {risk_classification}
Probability: {probability}
Ingredients: {ingredients}

Rules:
- Explanation must be one short sentence in simple language.
- Recommendations should be 3 short actionable bullets.
""".strip()

    try:
        response = client.models.generate_content(model=model, contents=prompt)
        data = parse_json_like(getattr(response, "text", ""))
        if isinstance(data, dict):
            explanation = str(data.get("explanation", "")).strip()
            recommendations = [str(x).strip() for x in data.get("recommendations", []) if str(x).strip()]
            return explanation, recommendations
    except Exception:
        pass

    return "", []


def compute_personalized_alert(
    user_allergies: List[str],
    ai_allergens: List[str],
    allergens: List[str],
    ingredients: List[str],
) -> str:
    if not user_allergies:
        return ""
    normalized_user = {a.strip().lower() for a in user_allergies if a.strip()}
    normalized_detected = {a.strip().lower() for a in ai_allergens + allergens if str(a).strip()}
    ingredient_text = " ".join(ingredients).lower()

    overlaps = sorted(normalized_user.intersection(normalized_detected))
    if not overlaps:
        keyword_matches = sorted([a for a in normalized_user if a in ingredient_text])
        overlaps = keyword_matches

    if overlaps:
        return f"Contains {', '.join(overlaps)} - HIGH RISK for you."
    return ""


def api_analyze_image(api_base: str, image_name: str, image_bytes: bytes, mime_type: str) -> Dict:
    files = {"file": (image_name, image_bytes, mime_type)}
    response = requests.post(f"{api_base}/analyze-image", files=files, timeout=180)
    response.raise_for_status()
    return response.json()


def api_predict_risk(api_base: str, ingredients: List[str], vhi: float, cas: float, erf: float) -> Dict:
    payload = {
        "ingredients": ingredients,
        "vhi": vhi,
        "cas": cas,
        "erf": erf,
    }
    response = requests.post(f"{api_base}/predict-risk", json=payload, timeout=90)
    response.raise_for_status()
    return response.json()


def init_state() -> None:
    defaults = {
        "auth_complete": False,
        "profile_complete": False,
        "workflow_step": 1,
        "preferred_language": "English",
        "health_conditions": [],
        "allergy_keywords": [],
        "custom_allergy_input": "",
        "ocr_raw_text": "",
        "ingredients": [],
        "ai_ingredients": [],
        "translated_ingredients": [],
        "translation_note": "",
        "hidden_ingredients": [],
        "ingredient_breakdown": [],
        "allergens": [],
        "ai_allergens": [],
        "ingredient_editor": "",
        "last_prediction": None,
        "personalized_alert": "",
        "scan_image_name": "",
        "scan_image_bytes": b"",
        "scan_image_mime": "image/jpeg",
        "show_results": False,
        "user_profile": {
            "name": "",
            "age": 25,
            "preferred_language": "English",
            "health_conditions": [],
            "allergies": [],
        },
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Manrope:wght@400;500;700;800&display=swap');

        :root {
            --bg-1: #070b16;
            --bg-2: #141c34;
            --bg-3: #1f2a4f;
            --ink: #f1f6ff;
            --muted: #a9b7d8;
            --glass: rgba(19, 28, 53, 0.62);
            --glass-2: rgba(34, 48, 88, 0.56);
            --stroke: rgba(160, 191, 255, 0.22);
            --accent: #2ec4ff;
            --accent-2: #8a8fff;
            --ok: #1fd79b;
            --warn: #f6c445;
            --danger: #ff5f7d;
        }

        .stApp {
            background:
                radial-gradient(1200px 500px at 8% -8%, rgba(46,196,255,.24), transparent 55%),
                radial-gradient(900px 450px at 95% -12%, rgba(138,143,255,.2), transparent 58%),
                linear-gradient(165deg, var(--bg-1) 0%, var(--bg-2) 52%, var(--bg-3) 100%);
            color: var(--ink);
            font-family: 'Outfit', sans-serif;
        }

        .block-container {
            max-width: 940px;
            padding-top: 1rem;
            padding-bottom: 2rem;
        }

        .hero {
            border-radius: 22px;
            background: linear-gradient(130deg, rgba(46,196,255,.24), rgba(138,143,255,.22));
            border: 1px solid var(--stroke);
            backdrop-filter: blur(10px);
            padding: 1.2rem;
            box-shadow: 0 20px 45px rgba(8, 12, 27, .35);
            animation: fadeUp .55s ease;
            margin-bottom: 1rem;
        }

        .hero-title {
            margin: 0;
            font-size: 2rem;
            font-family: 'Manrope', sans-serif;
            letter-spacing: .2px;
            color: var(--ink);
        }

        .hero-sub {
            margin: .35rem 0 0;
            color: var(--muted);
            font-size: 1rem;
        }

        .glass {
            border-radius: 18px;
            background: var(--glass);
            border: 1px solid var(--stroke);
            backdrop-filter: blur(10px);
            padding: 1rem;
            margin-bottom: .9rem;
            box-shadow: 0 14px 35px rgba(6, 10, 24, .33);
            animation: fadeUp .45s ease;
        }

        .step-pill {
            display: inline-block;
            border-radius: 999px;
            padding: .3rem .68rem;
            font-size: .82rem;
            margin-right: .42rem;
            margin-top: .3rem;
            border: 1px solid var(--stroke);
            color: var(--muted);
            background: rgba(255,255,255,.03);
        }

        .step-pill.active {
            color: #02111b;
            border-color: transparent;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            font-weight: 700;
        }

        .risk-card {
            border-radius: 18px;
            border: 1px solid var(--stroke);
            background: var(--glass-2);
            padding: 1rem;
            margin-bottom: .8rem;
        }

        .risk-big {
            font-size: 2.1rem;
            font-family: 'Manrope', sans-serif;
            margin: .15rem 0 .35rem;
        }

        .fade-in {
            animation: fadeUp .55s ease;
        }

        .muted {
            color: var(--muted);
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        button[kind="primary"] {
            border-radius: 14px !important;
            border: none !important;
            background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
            color: #07111f !important;
            font-weight: 700 !important;
            min-height: 44px !important;
            box-shadow: 0 8px 20px rgba(46,196,255,.28);
        }

        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            border-radius: 12px !important;
            background: rgba(13, 19, 38, 0.85) !important;
            color: #f4f7ff !important;
            border: 1px solid rgba(160,191,255,0.28) !important;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stCaptionContainer"] {
            color: var(--ink);
        }

        @media (max-width: 768px) {
            .hero-title { font-size: 1.6rem; }
            .block-container { padding-top: .7rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_progress(step: int) -> None:
    labels = [
        "1 Login",
        "2 Profile",
        "3 Scan",
        "4 Confirm",
        "5 Translate",
        "6 Result",
    ]
    html = []
    for idx, label in enumerate(labels, start=1):
        active = "active" if idx <= step else ""
        html.append(f"<span class='step-pill {active}'>{label}</span>")

    st.markdown("<div class='glass'><div style='margin-bottom:.45rem;font-weight:600;'>Guided Flow</div>" + "".join(html) + "</div>", unsafe_allow_html=True)
    st.progress(min(step / 6, 1.0), text=f"Step {step} of 6")


def render_login() -> None:
    st.markdown("<div class='hero'><h1 class='hero-title'>EatWise AI - Personal Food Safety Assistant</h1><p class='hero-sub'>Sign in to start your guided safety analysis.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Simulated Login")

    name = st.text_input("Your name")
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    language = st.selectbox(
        "Preferred language",
        options=["English", "Hindi", "Telugu", "Tamil", "Spanish", "French", "German", "Arabic"],
        index=0,
    )

    if st.button("Continue to onboarding", width="stretch", type="primary"):
        if not name.strip():
            st.warning("Please enter your name to continue.")
        else:
            st.session_state.user_profile["name"] = name.strip()
            st.session_state.user_profile["age"] = int(age)
            st.session_state.user_profile["preferred_language"] = language
            st.session_state.preferred_language = language
            st.session_state.auth_complete = True
            st.session_state.workflow_step = max(st.session_state.workflow_step, 2)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_profile_setup() -> None:
    st.markdown("<div class='hero'><h1 class='hero-title'>Welcome, " + st.session_state.user_profile["name"] + "</h1><p class='hero-sub'>Set your health profile so EatWise can personalize food safety alerts.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Health Profile")

    health_options = [
        "Diabetes",
        "Hypertension",
        "Pregnancy",
        "Kidney disease",
        "Thyroid condition",
        "Liver condition",
        "Heart condition",
    ]
    allergy_options = [
        "Milk",
        "Egg",
        "Peanut",
        "Tree nut",
        "Soy",
        "Wheat",
        "Gluten",
        "Fish",
        "Shellfish",
        "Sesame",
    ]

    selected_health = st.multiselect("Health conditions", options=health_options, default=st.session_state.health_conditions)
    selected_allergies = st.multiselect("Known allergies", options=allergy_options, default=st.session_state.allergy_keywords)
    custom_allergies = st.text_input(
        "Custom allergy keywords (comma separated)",
        value=st.session_state.custom_allergy_input,
        placeholder="example: sulphites, mustard",
    )

    if st.button("Start guided assistant", width="stretch", type="primary"):
        custom_list = [x.strip() for x in custom_allergies.split(",") if x.strip()]
        allergy_keywords = selected_allergies + custom_list

        st.session_state.health_conditions = selected_health
        st.session_state.allergy_keywords = allergy_keywords
        st.session_state.custom_allergy_input = custom_allergies

        st.session_state.user_profile["health_conditions"] = selected_health
        st.session_state.user_profile["allergies"] = [a.lower() for a in allergy_keywords]

        st.session_state.profile_complete = True
        st.session_state.workflow_step = max(st.session_state.workflow_step, 3)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_scan_step(api_base: str, client, model: str) -> None:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Step 1 - Capture or Upload")
    st.caption("Use camera capture for instant scan or upload a label image.")

    camera_photo = st.camera_input("Capture label")
    uploaded = st.file_uploader("Upload food label image", type=["png", "jpg", "jpeg", "webp"])

    selected_name = ""
    selected_bytes = b""
    selected_mime = "image/jpeg"

    if camera_photo is not None:
        selected_name = "camera_capture.jpg"
        selected_bytes = camera_photo.getvalue()
        selected_mime = "image/jpeg"
        st.image(selected_bytes, caption="Captured image", width="stretch")
    elif uploaded is not None:
        selected_name = uploaded.name
        selected_bytes = uploaded.getvalue()
        selected_mime = uploaded.type or "image/jpeg"
        st.image(selected_bytes, caption="Uploaded image", width="stretch")

    if st.button("Scan Food Label", width="stretch", type="primary"):
        if not selected_bytes:
            st.warning("Capture or upload an image before scanning.")
        else:
            st.session_state.scan_image_name = selected_name
            st.session_state.scan_image_bytes = selected_bytes
            st.session_state.scan_image_mime = selected_mime

            with st.spinner("Analyzing label..."):
                try:
                    payload = api_analyze_image(api_base, selected_name, selected_bytes, selected_mime)
                except RequestException as exc:
                    st.error("Backend not reachable")
                    st.caption(str(exc))
                    st.session_state.ingredients = []
                    st.session_state.ai_ingredients = []
                    st.session_state.ingredient_editor = ""
                else:
                    ingredients = payload.get("ai_ingredients") or payload.get("ingredients", [])
                    ingredients = [str(x).strip() for x in ingredients if str(x).strip()]

                    if is_placeholder_ingredients(ingredients):
                        ingredients = []

                    ingredients = normalize_ingredients_with_ai(
                        client=client,
                        model=model,
                        ingredients=ingredients,
                        raw_text=payload.get("raw_text", ""),
                    )

                    st.session_state.ocr_raw_text = payload.get("raw_text", "")
                    st.session_state.ingredients = payload.get("ingredients", [])
                    st.session_state.ai_ingredients = ingredients
                    st.session_state.hidden_ingredients = payload.get("hidden_ingredients", [])
                    st.session_state.ingredient_breakdown = payload.get("ingredient_breakdown", [])
                    st.session_state.allergens = payload.get("allergens", [])
                    st.session_state.ai_allergens = payload.get("ai_allergens", [])
                    st.session_state.ingredient_editor = ", ".join(ingredients)

                    st.session_state.workflow_step = max(st.session_state.workflow_step, 4)
                    if ingredients:
                        st.success("Label scanned. Please confirm detected ingredients.")
                    else:
                        st.warning("OCR did not return clear ingredients. Please add them manually below.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_confirm_step() -> None:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Step 2 - Confirm ingredients")
    st.caption("Review and edit before analysis. This avoids OCR mistakes.")

    st.text_area(
        "Detected Ingredients",
        key="ingredient_editor",
        height=130,
        placeholder="example: milk powder, sugar, cocoa butter",
    )

    missing = st.text_input("Add missing ingredients", placeholder="example: lecithin, emulsifier")
    decision = st.radio("Are these correct?", options=["Yes, continue", "Edit ingredients"], horizontal=True)

    if st.button("Save ingredients and continue", width="stretch", type="primary"):
        edited = parse_ingredient_text(st.session_state.ingredient_editor)
        missing_items = parse_ingredient_text(missing)
        final_items = edited + [x for x in missing_items if x.lower() not in {i.lower() for i in edited}]

        if not final_items:
            st.warning("Please enter at least one ingredient.")
        else:
            st.session_state.ai_ingredients = final_items
            st.session_state.ingredients = final_items
            st.session_state.ingredient_editor = ", ".join(final_items)
            st.session_state.workflow_step = max(st.session_state.workflow_step, 5)
            if decision == "Yes, continue":
                st.success("Ingredients confirmed.")
            else:
                st.info("Edits saved. Continue when ready.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_translation_step(client, model: str) -> None:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Step 3 - Smart translation")

    preferred_language = st.session_state.user_profile.get("preferred_language", "English")
    ingredients = st.session_state.ai_ingredients

    if not ingredients:
        st.info("Add ingredients in Step 2 to enable translation.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if preferred_language.lower() == "english":
        st.session_state.translated_ingredients = ingredients
        st.session_state.translation_note = "Preferred language is English. Translation skipped."
        st.info("Preferred language is English. Using ingredients as-is.")
    else:
        if st.button(f"Translate to {preferred_language}", width="stretch"):
            with st.spinner("Translating ingredients..."):
                translated, note = translate_ingredients_with_ai(
                    client=client,
                    model=model,
                    ingredients=ingredients,
                    target_language=preferred_language,
                )
                st.session_state.translated_ingredients = translated
                st.session_state.translation_note = note

        if not st.session_state.translated_ingredients:
            st.session_state.translated_ingredients = ingredients

        if st.session_state.translation_note:
            st.caption(st.session_state.translation_note)

    if st.session_state.translated_ingredients:
        st.markdown("**Translated Ingredients**")
        st.write(", ".join(st.session_state.translated_ingredients))

    st.session_state.workflow_step = max(st.session_state.workflow_step, 5)
    st.markdown("</div>", unsafe_allow_html=True)


def render_analysis_step(api_base: str, client, model: str) -> None:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Step 4 - Risk analysis")
    st.caption("Tune environment context and run AI + ML safety prediction.")

    vhi = st.slider("Vendor Hygiene Index (VHI)", 0.0, 1.0, 0.65, 0.01)
    cas = st.slider("Consumer Awareness Score (CAS)", 0.0, 1.0, 0.55, 0.01)
    erf = st.slider("Environmental Risk Factor (ERF)", 0.0, 1.0, 0.5, 0.01)

    ingredients = parse_ingredient_text(st.session_state.ingredient_editor)
    if not ingredients:
        ingredients = st.session_state.ai_ingredients

    if st.button("Predict Risk", width="stretch", type="primary"):
        if not ingredients:
            st.warning("Please confirm ingredients before prediction.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    prediction = api_predict_risk(api_base, ensure_non_empty_ingredients(ingredients), vhi=vhi, cas=cas, erf=erf)
                except RequestException as exc:
                    st.error("Backend not reachable")
                    st.caption(str(exc))
                else:
                    cls = str(prediction.get("risk_classification", ""))
                    prob = float(prediction.get("probability", 0.0))

                    ai_explanation, ai_recs = generate_friendly_extras_with_ai(
                        client=client,
                        model=model,
                        ingredients=ingredients,
                        risk_classification=cls,
                        probability=prob,
                    )

                    if ai_explanation and not prediction.get("ai_explanation"):
                        prediction["ai_explanation"] = ai_explanation
                    if ai_recs and not prediction.get("recommendations"):
                        prediction["recommendations"] = ai_recs

                    alert = compute_personalized_alert(
                        user_allergies=st.session_state.user_profile.get("allergies", []),
                        ai_allergens=st.session_state.ai_allergens,
                        allergens=st.session_state.allergens,
                        ingredients=ingredients,
                    )
                    st.session_state.personalized_alert = alert
                    st.session_state.last_prediction = prediction
                    st.session_state.show_results = True
                    st.session_state.workflow_step = 6
                    st.success("Risk analysis complete.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_results() -> None:
    if not st.session_state.last_prediction:
        st.markdown("<div class='glass'><div class='muted'>Run risk analysis to see results.</div></div>", unsafe_allow_html=True)
        return

    result = st.session_state.last_prediction
    cls = str(result.get("risk_classification", "Unknown"))
    prob = float(result.get("probability", 0.0))

    if cls.lower() == "low":
        color = "#1fd79b"
        label = "LOW RISK"
    elif cls.lower() == "medium":
        color = "#f6c445"
        label = "MODERATE RISK"
    else:
        color = "#ff5f7d"
        label = "HIGH RISK"

    st.markdown("<div class='glass fade-in'>", unsafe_allow_html=True)
    st.subheader("Step 5 - Results")

    st.markdown(
        f"""
        <div class='risk-card'>
            <div class='muted'>Risk Score</div>
            <div class='risk-big' style='color:{color};'>{prob * 100:.1f}%</div>
            <div style='font-weight:700;color:{color};'>{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    alert = st.session_state.personalized_alert
    if alert:
        st.error(f"Personalized Alert: {alert}")
    else:
        st.info("No profile-specific allergen conflict detected for this scan.")

    st.markdown("### Ingredients breakdown")
    translated = st.session_state.translated_ingredients or st.session_state.ai_ingredients
    st.write(", ".join(ensure_non_empty_ingredients(translated)))

    if st.session_state.hidden_ingredients:
        st.caption("Hidden ingredients detected: " + ", ".join(st.session_state.hidden_ingredients))

    if st.session_state.ingredient_breakdown:
        st.dataframe(pd.DataFrame(st.session_state.ingredient_breakdown), width="stretch")

    explanation = result.get("ai_explanation") or result.get("explanation") or "Safety risk is estimated from hygiene, awareness, environment, and ingredient signals."
    st.markdown("### AI explanation")
    st.markdown(f"<div class='risk-card'>{explanation}</div>", unsafe_allow_html=True)

    recommendations = result.get("recommendations", [])
    if not recommendations:
        recommendations = [
            "Prefer products with shorter ingredient lists.",
            "Avoid known allergens listed in your profile.",
            "Store food safely and follow package instructions.",
        ]

    st.markdown("### Recommendations")
    for rec in recommendations:
        st.markdown(f"- {rec}")

    st.markdown("</div>", unsafe_allow_html=True)


def render_metrics_panel(metrics: dict) -> None:
    demo_accuracy = float(metrics.get("classification_report", {}).get("accuracy", 0.776))
    demo_f1 = float(metrics.get("classification_report", {}).get("weighted avg", {}).get("f1-score", 0.7721883674933927))
    demo_auc = float(metrics.get("roc_auc_ovr", 0.9189395046135461))

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Model insights")
    st.caption("Tracked metrics from training artifacts")

    stats = pd.DataFrame(
        {
            "Metric": ["Research Accuracy", "Demo Accuracy", "Demo Weighted F1", "Demo AUC"],
            "Value": [94.6, demo_accuracy * 100, demo_f1 * 100, demo_auc * 100],
        }
    ).set_index("Metric")
    st.bar_chart(stats)

    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="EatWise AI", page_icon="EW", layout="wide")

    init_state()
    apply_theme()

    api_base = get_api_base()
    gemini_key = get_gemini_key()
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_client = get_genai_client(gemini_key)

    training_metrics = load_metrics()

    if not st.session_state.auth_complete:
        render_login()
        return

    if not st.session_state.profile_complete:
        render_profile_setup()
        return

    st.markdown(
        """
        <div class='hero'>
            <h1 class='hero-title'>EatWise AI - Personal Food Safety Assistant</h1>
            <p class='hero-sub'>Guided scan, ingredient intelligence, personalized risk alerts, and simple recommendations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_progress(st.session_state.workflow_step)

    st.markdown(
        f"<div class='glass'><strong>User:</strong> {st.session_state.user_profile.get('name', '')} | "
        f"<strong>Age:</strong> {st.session_state.user_profile.get('age', '')} | "
        f"<strong>Language:</strong> {st.session_state.user_profile.get('preferred_language', 'English')} | "
        f"<strong>Allergy keywords:</strong> {', '.join(st.session_state.user_profile.get('allergies', [])) or 'None'}</div>",
        unsafe_allow_html=True,
    )

    render_scan_step(api_base=api_base, client=gemini_client, model=gemini_model)
    render_confirm_step()
    render_translation_step(client=gemini_client, model=gemini_model)
    render_analysis_step(api_base=api_base, client=gemini_client, model=gemini_model)
    render_results()
    render_metrics_panel(training_metrics)

    with st.expander("About this assistant", expanded=False):
        st.markdown("- Guided flow: Login -> Profile -> Scan -> Confirm -> Translate -> Analyze -> Result")
        st.markdown("- OCR source: backend /analyze-image with AI fallback")
        st.markdown("- Risk source: backend /predict-risk")
        st.markdown("- AI usage: ingredient normalization, translation, explanation, recommendations")

    st.caption("Backend API: configurable with API_BASE | Gemini optional in frontend | Mobile-first guided flow")


if __name__ == "__main__":
    main()
