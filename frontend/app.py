from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Tuple

import requests
import streamlit as st
from requests import RequestException

STEP_LABELS = [
    "Profile",
    "Scan",
    "Ingredients",
    "Clean & Translate",
    "Analyze",
    "Results",
]

LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Arabic": "ar",
}

RISK_MAP = {
    "Good": 0.2,
    "Average": 0.5,
    "Poor": 0.85,
    "Not sure": 0.55,
}

NOISE_WORDS = {
    "fer",
    "who",
    "mac",
    "nan",
    "ingred",
    "bahi",
    "ingredient",
    "ingredients",
    "contains",
    "nutrition",
    "nutritional",
    "may",
    "contain",
}


def get_api_base() -> str:
    try:
        secret_api_base = st.secrets.get("API_BASE")
    except Exception:
        secret_api_base = None
    return os.getenv("API_BASE") or secret_api_base or "https://eatwise-ai.onrender.com"


def parse_ingredient_text(value: str) -> List[str]:
    seen = set()
    parsed: List[str] = []
    normalized_value = value.replace(";", ",").replace("\n", ",")
    for item in re.split(r"(?<!\d),(?!\d)", normalized_value):
        cleaned = item.strip()
        lowered = cleaned.lower()
        if cleaned and lowered not in seen:
            parsed.append(cleaned)
            seen.add(lowered)
    return parsed


def clean_ingredients_for_ui(items: List[str]) -> List[str]:
    cleaned: List[str] = []
    seen = set()

    for raw in items:
        token = str(raw).strip().lower()
        token = re.sub(r"\b\d+(?:[.,]\d+)?%?\b", " ", token)
        token = re.sub(r"[^a-z\s\-]", " ", token)
        token = re.sub(r"\s+", " ", token).strip(" .,-")

        if not token or len(token) < 3:
            continue
        if token in NOISE_WORDS:
            continue
        if not re.fullmatch(r"[a-z\-\s]+", token):
            continue
        if len(token.split()) > 6:
            continue

        if "lecithin" in token and ("soya" in token or "soy" in token):
            token = "lecithin"
        elif token == "milk powder":
            token = "milk"
        elif token == "wheat flour":
            token = "wheat"

        normalized = token.strip().lower()
        if normalized in seen:
            continue

        seen.add(normalized)
        cleaned.append(normalized)
        if len(cleaned) >= 30:
            break

    return cleaned


def ingredient_signature(ingredients: List[str], language: str) -> str:
    return "|".join(ingredients) + f"::{language.lower()}"


def compute_personalized_alert(user_allergies: List[str], allergens_detected: List[str], ingredients: List[str]) -> str:
    if not user_allergies:
        return ""

    normalized_user = {x.strip().lower() for x in user_allergies if x.strip()}
    normalized_detected = {x.strip().lower() for x in allergens_detected if str(x).strip()}
    ingredient_text = " ".join(ingredients).lower()

    overlap = sorted(normalized_user.intersection(normalized_detected))
    if not overlap:
        overlap = sorted([a for a in normalized_user if a in ingredient_text])

    if overlap:
        return f"Personal alert: contains or may contain {', '.join(overlap)}."
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


def translate_with_libretranslate(ingredients: List[str], target_language_name: str) -> Tuple[List[str], str]:
    if not ingredients:
        return [], "No ingredients to translate."

    if target_language_name.lower() == "english":
        return ingredients, "Language is English. Translation skipped."

    target_code = LANGUAGE_CODES.get(target_language_name, "")
    if not target_code:
        return ingredients, "Translation unavailable for selected language."

    text_payload = "\n".join(ingredients)
    endpoints = [
        "https://translate.argosopentech.com/translate",
        "https://libretranslate.de/translate",
    ]

    for endpoint in endpoints:
        try:
            response = requests.post(
                endpoint,
                json={
                    "q": text_payload,
                    "source": "en",
                    "target": target_code,
                    "format": "text",
                },
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            translated_text = str(data.get("translatedText", "")).strip()
            if not translated_text:
                continue

            translated = [line.strip() for line in translated_text.split("\n") if line.strip()]
            if translated:
                return translated, f"Translated to {target_language_name} using LibreTranslate."
        except Exception:
            continue

    return ingredients, "Translation unavailable right now. Showing English ingredients."


def map_risk_inputs() -> Tuple[float, float, float]:
    return (
        float(RISK_MAP[st.session_state.hygiene_choice]),
        float(RISK_MAP[st.session_state.awareness_choice]),
        float(RISK_MAP[st.session_state.storage_choice]),
    )


def init_state() -> None:
    defaults = {
        "step": 1,
        "profile": {
            "name": "",
            "age": 25,
            "preferred_language": "English",
            "health_conditions": [],
            "allergies": [],
        },
        "scan_source": "upload",
        "scan_image_name": "",
        "scan_image_bytes": b"",
        "scan_image_mime": "image/jpeg",
        "ocr_source": "fallback",
        "ingredients": [],
        "display_ingredients": [],
        "ingredient_editor": "",
        "translation_note": "",
        "processing_signature": "",
        "hygiene_choice": "Good",
        "awareness_choice": "Good",
        "storage_choice": "Good",
        "prediction": None,
        "prediction_signature": "",
        "personal_alert": "",
        "allergens": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #ffffff;
            --surface: #ffffff;
            --surface-soft: #f7f8fc;
            --ink: #0f172a;
            --muted: #64748b;
            --stroke: #e7eaf2;
            --accent: #6d5efc;
            --accent-2: #28b37d;
            --accent-soft: rgba(109, 94, 252, 0.12);
            --shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
            --shadow-soft: 0 8px 20px rgba(15, 23, 42, 0.06);
            --ok: #28b37d;
            --warn: #d39b17;
            --danger: #ef5f6c;
        }

        .stApp {
            background: var(--bg);
            color: var(--ink);
            font-family: 'Inter', sans-serif;
        }

        .block-container {
            max-width: 960px;
            padding-top: 1rem;
            padding-bottom: 2rem;
        }

        .hero {
            background: linear-gradient(180deg, #ffffff 0%, #fbfbfe 100%);
            border: 1px solid var(--stroke);
            border-radius: 22px;
            padding: 1.15rem 1.15rem 1rem;
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
            animation: fadeUp .45s ease;
        }

        .hero h1 {
            margin: 0;
            font-size: 1.85rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: var(--ink);
        }

        .hero p {
            margin: .35rem 0 0;
            color: var(--muted);
            line-height: 1.5;
        }

        .panel {
            background: var(--surface);
            border: 1px solid var(--stroke);
            border-radius: 16px;
            box-shadow: var(--shadow-soft);
            padding: 1rem;
            margin-bottom: .9rem;
            animation: fadeUp .38s ease;
        }

        .subtle {
            color: var(--muted);
        }

        .step-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            padding: .45rem .75rem;
            margin: .18rem .28rem .18rem 0;
            border: 1px solid var(--stroke);
            background: #fff;
            color: var(--muted);
            font-size: .84rem;
            font-weight: 600;
            transition: all .25s ease;
        }

        .step-pill.active {
            background: linear-gradient(135deg, rgba(109,94,252,.12), rgba(40,179,125,.12));
            border-color: rgba(109,94,252,.25);
            color: var(--ink);
            transform: translateY(-1px);
        }

        .step-track {
            display: flex;
            flex-wrap: wrap;
            gap: .15rem;
        }

        .section-tag {
            font-size: .78rem;
            letter-spacing: .08em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: .45rem;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: .42rem .75rem;
            margin: .2rem .3rem .2rem 0;
            background: #f8f9fd;
            border: 1px solid var(--stroke);
            color: var(--ink);
            font-size: .86rem;
            box-shadow: 0 1px 0 rgba(15,23,42,.02);
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: .75rem;
        }

        .stat-card {
            background: #fff;
            border: 1px solid var(--stroke);
            border-radius: 16px;
            padding: .85rem;
            box-shadow: var(--shadow-soft);
        }

        .risk-card {
            border-radius: 18px;
            padding: 1rem;
            border: 1px solid var(--stroke);
            background: linear-gradient(180deg, #ffffff 0%, #fbfbff 100%);
            box-shadow: var(--shadow);
            text-align: center;
        }

        .risk-score {
            font-size: 2.7rem;
            font-weight: 800;
            line-height: 1;
            margin: .15rem 0 .3rem;
            letter-spacing: -0.04em;
        }

        .result-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: .75rem;
        }

        .how-card {
            background: #fff;
            border: 1px solid var(--stroke);
            border-radius: 16px;
            padding: .95rem;
            box-shadow: var(--shadow-soft);
        }

        .ai-card {
            border-radius: 16px;
            padding: .95rem;
            background: #fbfbff;
            border: 1px solid #ebe8ff;
            box-shadow: var(--shadow-soft);
        }

        button[kind="primary"] {
            border-radius: 14px !important;
            border: none !important;
            background: linear-gradient(135deg, var(--accent), #8c7bff) !important;
            color: #fff !important;
            font-weight: 700 !important;
            min-height: 44px !important;
            box-shadow: 0 10px 22px rgba(109,94,252,.2);
        }

        button[kind="secondary"] {
            border-radius: 14px !important;
            background: #fff !important;
            border: 1px solid var(--stroke) !important;
            color: var(--ink) !important;
            min-height: 44px !important;
        }

        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            border-radius: 12px !important;
            background: #fff !important;
            color: var(--ink) !important;
            border: 1px solid var(--stroke) !important;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stCaptionContainer"] {
            color: var(--ink);
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 768px) {
            .grid-2,
            .result-grid {
                grid-template-columns: 1fr;
            }

            .hero h1 {
                font-size: 1.55rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_progress(step: int) -> None:
    pills = []
    for idx, label in enumerate(STEP_LABELS, start=1):
        klass = "active" if idx <= step else ""
        pills.append(f"<span class='step-pill {klass}'>Step {idx}: {label}</span>")

    st.markdown(
        "<div class='panel'><div class='section-tag'>Step Progress</div><div class='step-track'>"
        + "".join(pills)
        + "</div></div>",
        unsafe_allow_html=True,
    )
    st.progress(min(step / 6, 1.0), text=f"Step {step} of 6")


def render_chips(items: List[str]) -> None:
    if not items:
        st.caption("No ingredients available yet.")
        return
    chips = "".join([f"<span class='chip'>{item}</span>" for item in items])
    st.markdown(chips, unsafe_allow_html=True)


def step_jump(target_step: int) -> None:
    time.sleep(0.14)
    st.session_state.step = target_step
    st.rerun()


def render_step_1() -> None:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Step 1 - Profile")
    st.caption("Enter your profile to personalize the experience.")

    name = st.text_input("Name", value=st.session_state.profile.get("name", ""), placeholder="Your name")
    age = st.number_input("Age", min_value=1, max_value=120, value=int(st.session_state.profile.get("age", 25)))
    allergies = st.multiselect(
        "Allergies",
        options=["milk", "egg", "peanut", "tree nut", "soy", "wheat", "gluten", "fish", "shellfish", "sesame"],
        default=st.session_state.profile.get("allergies", []),
    )
    preferred_language = st.selectbox(
        "Preferred language",
        options=list(LANGUAGE_CODES.keys()),
        index=list(LANGUAGE_CODES.keys()).index(st.session_state.profile.get("preferred_language", "English")),
    )

    if st.button("Next", type="primary", use_container_width=True):
        if not name.strip():
            st.warning("Please enter your name.")
        else:
            st.session_state.profile = {
                "name": name.strip(),
                "age": int(age),
                "preferred_language": preferred_language,
                "health_conditions": [],
                "allergies": [a.lower() for a in allergies],
            }
            step_jump(2)

    st.markdown("</div>", unsafe_allow_html=True)


def render_step_2(api_base: str) -> None:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Step 2 - Scan")
    st.caption("Upload a label image or use the camera.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Upload Image", type="secondary", use_container_width=True):
            st.session_state.scan_source = "upload"
    with col2:
        if st.button("Use Camera", type="secondary", use_container_width=True):
            st.session_state.scan_source = "camera"

    selected_name = ""
    selected_bytes = b""
    selected_mime = "image/jpeg"

    if st.session_state.scan_source == "camera":
        camera_photo = st.camera_input("Use Camera")
        if camera_photo is not None:
            selected_name = "camera_capture.jpg"
            selected_bytes = camera_photo.getvalue()
            st.image(selected_bytes, caption="Captured image", use_container_width=True)
    else:
        uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])
        if uploaded is not None:
            selected_name = uploaded.name
            selected_bytes = uploaded.getvalue()
            selected_mime = uploaded.type or "image/jpeg"
            st.image(selected_bytes, caption="Uploaded image", use_container_width=True)

    if st.button("Scan Label", type="primary", use_container_width=True):
        if not selected_bytes:
            st.warning("Please upload or capture an image first.")
        else:
            st.session_state.scan_image_name = selected_name
            st.session_state.scan_image_bytes = selected_bytes
            st.session_state.scan_image_mime = selected_mime

            with st.spinner("Scanning your food label..."):
                st.progress(15, text="Starting OCR...")
                time.sleep(0.25)
                try:
                    payload = api_analyze_image(api_base, selected_name, selected_bytes, selected_mime)
                except RequestException:
                    st.error("Unable to process. Please try again.")
                    return

            ingredients = clean_ingredients_for_ui([str(x) for x in (payload.get("ai_ingredients") or payload.get("ingredients", []))])
            st.session_state.ingredients = ingredients
            st.session_state.display_ingredients = ingredients
            st.session_state.ingredient_editor = ", ".join(ingredients)
            st.session_state.ocr_source = str(payload.get("source", "fallback"))
            st.session_state.allergens = [str(x).lower() for x in payload.get("allergens", []) if str(x).strip()]
            st.session_state.prediction = None
            st.session_state.prediction_signature = ""
            step_jump(3)

    st.markdown("</div>", unsafe_allow_html=True)


def render_step_3() -> None:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Step 3 - Ingredients")
    st.caption("Extracted Ingredients")

    ingredients = st.session_state.ingredients
    if ingredients:
        for idx, item in enumerate(ingredients):
            time.sleep(0.01)
            st.markdown(f"<div class='chip'>{item}</div>", unsafe_allow_html=True)
    else:
        st.info("No cleaned ingredients are available yet.")

    st.text_area(
        "Editable text",
        key="ingredient_editor",
        height=140,
        placeholder="milk, sugar, cocoa butter",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back", type="secondary", use_container_width=True):
            step_jump(2)
    with col2:
        if st.button("Next", type="primary", use_container_width=True):
            edited = clean_ingredients_for_ui(parse_ingredient_text(st.session_state.ingredient_editor))
            if not edited:
                st.warning("Please provide at least one valid ingredient.")
            else:
                st.session_state.ingredients = edited
                st.session_state.display_ingredients = edited
                step_jump(4)

    st.markdown("</div>", unsafe_allow_html=True)


def render_step_4() -> None:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Step 4 - Clean & Translate")

    preferred_language = st.session_state.profile.get("preferred_language", "English")
    signature = ingredient_signature(st.session_state.ingredients, preferred_language)

    with st.spinner("Cleaning and optimizing ingredients..."):
        st.progress(35, text="Cleaning ingredient list...")
        time.sleep(0.35)
        cleaned = clean_ingredients_for_ui(st.session_state.ingredients)
        translated, note = translate_with_libretranslate(cleaned, preferred_language)
        st.session_state.ingredients = cleaned
        st.session_state.display_ingredients = translated
        st.session_state.translation_note = note
        st.session_state.processing_signature = signature

    st.markdown("<div class='section-tag'>Cleaned Ingredients</div>", unsafe_allow_html=True)
    render_chips(st.session_state.ingredients)

    if preferred_language.lower() != "english":
        st.markdown("<div class='section-tag'>Translated Ingredients</div>", unsafe_allow_html=True)
        render_chips(st.session_state.display_ingredients)

    if st.session_state.translation_note:
        st.caption(st.session_state.translation_note)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back", type="secondary", use_container_width=True):
            step_jump(3)
    with col2:
        if st.button("Next", type="primary", use_container_width=True):
            step_jump(5)

    st.markdown("</div>", unsafe_allow_html=True)


def render_step_5() -> None:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Step 5 - Analyze")
    st.caption("Map the friendly choices to the hybrid ML model inputs.")

    options = ["Good", "Average", "Poor", "Not sure"]

    st.session_state.hygiene_choice = st.radio("Hygiene", options=options, horizontal=True, index=options.index(st.session_state.hygiene_choice))
    st.session_state.awareness_choice = st.radio("Awareness", options=options, horizontal=True, index=options.index(st.session_state.awareness_choice))
    st.session_state.storage_choice = st.radio("Storage", options=options, horizontal=True, index=options.index(st.session_state.storage_choice))

    if st.button("Run Safety Model", type="primary", use_container_width=True):
        step_jump(6)

    st.markdown("</div>", unsafe_allow_html=True)


def render_result_cards(result: Dict, ingredients: List[str], translated_ingredients: List[str]) -> None:
    risk_class = str(result.get("risk_classification", "Unknown"))
    risk_score = float(result.get("risk_score", result.get("probability", 0.0))) * 100.0

    if risk_class.lower() == "low":
        risk_color = "var(--ok)"
        risk_bg = "rgba(40, 179, 125, 0.10)"
    elif risk_class.lower() == "medium":
        risk_color = "var(--warn)"
        risk_bg = "rgba(211, 155, 23, 0.10)"
    else:
        risk_color = "var(--danger)"
        risk_bg = "rgba(239, 95, 108, 0.10)"

    score_placeholder = st.empty()
    for value in range(0, int(risk_score) + 1, max(1, int(max(risk_score, 1) / 15))):
        score_placeholder.markdown(
            f"""
            <div class='risk-card' style='background: linear-gradient(180deg, #fff 0%, {risk_bg} 100%);'>
                <div class='section-tag' style='margin-bottom:.25rem;'>Risk Card</div>
                <div class='risk-score' style='color:{risk_color};'>{value:.0f}%</div>
                <div style='font-weight:700;color:{risk_color};'>{risk_class.upper()} RISK</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(0.02)

    score_placeholder.markdown(
        f"""
        <div class='risk-card' style='background: linear-gradient(180deg, #fff 0%, {risk_bg} 100%);'>
            <div class='section-tag' style='margin-bottom:.25rem;'>Risk Card</div>
            <div class='risk-score' style='color:{risk_color};'>{risk_score:.1f}%</div>
            <div style='font-weight:700;color:{risk_color};'>{risk_class.upper()} RISK</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    features = result.get("features", {}) if isinstance(result.get("features", {}), dict) else {}
    ingredient_risk = float(features.get("ingredient_risk", 0.0))
    erf_value = float(features.get("erf", 0.0))
    allergens_detected = [str(x).lower() for x in result.get("allergens_detected", []) if str(x).strip()]

    st.markdown("<div class='section-tag'>Model Explanation</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='stat-card'><div class='subtle'>Ingredient risk</div><h3>{ingredient_risk:.2f}</h3></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='stat-card'><div class='subtle'>ERF</div><h3>{erf_value:.2f}</h3></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='stat-card'><div class='subtle'>Allergen count</div><h3>{len(allergens_detected)}</h3></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-tag'>AI Explanation</div>", unsafe_allow_html=True)
    ai_text = str(result.get("ai_explanation", "")).strip() or str(result.get("explanation", "")).strip() or "No AI explanation available."
    st.markdown(f"<div class='ai-card'><strong>AI-generated explanation</strong><br>{ai_text}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-tag'>Ingredients</div>", unsafe_allow_html=True)
    render_chips(translated_ingredients if st.session_state.profile.get("preferred_language", "English").lower() != "english" else ingredients)

    if st.session_state.profile.get("preferred_language", "English").lower() != "english":
        st.caption("Translated list shown for the selected language.")

    st.markdown("<div class='section-tag'>Alert</div>", unsafe_allow_html=True)
    alert = st.session_state.personal_alert
    if alert:
        st.error(alert)
    else:
        st.success("No profile-specific allergen conflict detected.")


def render_how_it_works() -> None:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("How the model works")
    st.markdown(
        """
        <div class='how-card'>
            <p><strong>OCR extracts text</strong> from the food label image.</p>
            <p><strong>Rule-based cleaning</strong> removes noise, deduplicates, and normalizes ingredients.</p>
            <p><strong>ML model</strong> combines RF + GB + SVM signals with VHI, CAS, and ERF inputs.</p>
            <p><strong>SHAP explainability</strong> helps show why the model produced the prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_step_6(api_base: str) -> None:
    ingredients = st.session_state.ingredients
    translated_ingredients = st.session_state.display_ingredients or ingredients

    if not ingredients:
        st.error("No ingredients available. Please scan and clean ingredients first.")
        return

    vhi, cas, erf = map_risk_inputs()
    signature = f"{'|'.join(ingredients)}::{vhi:.2f}:{cas:.2f}:{erf:.2f}"

    if st.session_state.prediction is None or st.session_state.prediction_signature != signature:
        with st.spinner("Running safety model..."):
            st.progress(45, text="Preparing model input...")
            time.sleep(0.3)
            try:
                result = api_predict_risk(api_base=api_base, ingredients=ingredients, vhi=vhi, cas=cas, erf=erf)
            except RequestException:
                st.error("Unable to process. Please try again.")
                return

        allergens_detected = [str(x).lower() for x in result.get("allergens_detected", []) if str(x).strip()]
        st.session_state.personal_alert = compute_personalized_alert(
            user_allergies=st.session_state.profile.get("allergies", []),
            allergens_detected=allergens_detected,
            ingredients=ingredients,
        )
        st.session_state.prediction = result
        st.session_state.prediction_signature = signature

    render_result_cards(st.session_state.prediction, ingredients, translated_ingredients)
    render_how_it_works()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back", type="secondary", use_container_width=True):
            step_jump(5)
    with col2:
        if st.button("Analyze Another Label", type="primary", use_container_width=True):
            st.session_state.step = 2
            st.session_state.ingredients = []
            st.session_state.display_ingredients = []
            st.session_state.ingredient_editor = ""
            st.session_state.prediction = None
            st.session_state.prediction_signature = ""
            st.session_state.translation_note = ""
            st.session_state.personal_alert = ""
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="EatWise AI Assistant", page_icon="EW", layout="wide")
    init_state()
    apply_theme()

    api_base = get_api_base()

    st.markdown(
        """
        <div class='hero'>
            <h1>EatWise AI Assistant</h1>
            <p>A clean step-by-step assistant for food label safety analysis with a white, mobile-like guided experience.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_progress(st.session_state.step)

    if st.session_state.step == 1:
        render_step_1()
    elif st.session_state.step == 2:
        render_step_2(api_base)
    elif st.session_state.step == 3:
        render_step_3()
    elif st.session_state.step == 4:
        render_step_4()
    elif st.session_state.step == 5:
        render_step_5()
    else:
        render_step_6(api_base)


if __name__ == "__main__":
    main()
