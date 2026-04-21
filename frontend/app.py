import os
import re
from typing import Dict, List, Tuple

import requests
import streamlit as st
from requests import RequestException

STEP_LABELS = [
    "Welcome/Profile",
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
        float(RISK_MAP[st.session_state.environment_choice]),
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
        "environment_choice": "Good",
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
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Manrope:wght@500;700;800&display=swap');

        :root {
            --bg-1: #05070f;
            --bg-2: #0f1730;
            --bg-3: #1a2450;
            --ink: #f2f6ff;
            --muted: #a9b6d8;
            --glass: rgba(20, 28, 55, 0.62);
            --glass-2: rgba(36, 48, 88, 0.55);
            --stroke: rgba(176, 198, 255, 0.24);
            --ok: #21d29a;
            --warn: #f3c34f;
            --danger: #ff627b;
            --accent-1: #34d3ff;
            --accent-2: #8ea1ff;
        }

        .stApp {
            background:
                radial-gradient(900px 500px at 0% -8%, rgba(52,211,255,.2), transparent 60%),
                radial-gradient(900px 520px at 100% -10%, rgba(142,161,255,.18), transparent 62%),
                linear-gradient(160deg, var(--bg-1) 0%, var(--bg-2) 52%, var(--bg-3) 100%);
            color: var(--ink);
            font-family: 'Outfit', sans-serif;
        }

        .block-container {
            max-width: 980px;
            padding-top: 1rem;
            padding-bottom: 2rem;
        }

        .hero {
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(52,211,255,.2), rgba(142,161,255,.2));
            border: 1px solid var(--stroke);
            backdrop-filter: blur(10px);
            padding: 1.1rem;
            box-shadow: 0 18px 38px rgba(7, 12, 30, .4);
            animation: fadeUp .45s ease;
            margin-bottom: .9rem;
        }

        .glass {
            border-radius: 18px;
            border: 1px solid var(--stroke);
            background: var(--glass);
            backdrop-filter: blur(10px);
            padding: 1rem;
            margin-bottom: .85rem;
            box-shadow: 0 12px 32px rgba(6, 10, 26, .32);
            animation: fadeUp .38s ease;
        }

        .step-chip {
            display: inline-block;
            border-radius: 999px;
            padding: .28rem .72rem;
            margin: .2rem .28rem .1rem 0;
            border: 1px solid var(--stroke);
            color: var(--muted);
            font-size: .82rem;
            background: rgba(255,255,255,.03);
        }

        .step-chip.active {
            color: #02131e;
            border-color: transparent;
            background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
            font-weight: 700;
        }

        .ingredient-chip {
            display: inline-block;
            border-radius: 999px;
            padding: .34rem .68rem;
            margin: .22rem .28rem .22rem 0;
            border: 1px solid rgba(176,198,255,.28);
            background: rgba(255,255,255,.06);
            font-size: .84rem;
        }

        .section-label {
            font-size: .8rem;
            text-transform: uppercase;
            letter-spacing: .8px;
            color: var(--muted);
            margin-bottom: .35rem;
        }

        .risk-box {
            border-radius: 16px;
            border: 1px solid var(--stroke);
            background: var(--glass-2);
            padding: .95rem;
            margin-bottom: .7rem;
        }

        .risk-score {
            font-family: 'Manrope', sans-serif;
            font-size: 2.2rem;
            font-weight: 800;
            margin: .15rem 0;
        }

        .how-card {
            border-radius: 16px;
            border: 1px solid var(--stroke);
            background: rgba(255,255,255,.04);
            padding: .9rem;
        }

        button[kind="primary"] {
            border-radius: 13px !important;
            border: none !important;
            background: linear-gradient(135deg, #34d3ff, #21d29a) !important;
            color: #04141d !important;
            font-weight: 700 !important;
            min-height: 44px !important;
        }

        button[kind="secondary"] {
            border-radius: 13px !important;
            background: rgba(255,255,255,.08) !important;
            border: 1px solid rgba(176,198,255,.3) !important;
            color: var(--ink) !important;
            min-height: 44px !important;
        }

        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            border-radius: 12px !important;
            background: rgba(11, 17, 36, .86) !important;
            color: var(--ink) !important;
            border: 1px solid rgba(176,198,255,.28) !important;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stCaptionContainer"] {
            color: var(--ink);
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 768px) {
            .risk-score { font-size: 1.85rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_progress(step: int) -> None:
    pills = []
    for idx, label in enumerate(STEP_LABELS, start=1):
        klass = "active" if idx <= step else ""
        pills.append(f"<span class='step-chip {klass}'>Step {idx}: {label}</span>")

    st.markdown(
        "<div class='glass'><div style='font-weight:600;margin-bottom:.35rem;'>Assistant Flow</div>"
        + "".join(pills)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.progress(min(step / 6, 1.0), text=f"Step {step} of 6")


def render_ingredient_chips(ingredients: List[str]) -> None:
    if not ingredients:
        st.caption("No ingredients available yet.")
        return

    chips = "".join([f"<span class='ingredient-chip'>{item}</span>" for item in ingredients])
    st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)


def render_step_1() -> None:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Step 1 - Welcome/Profile")
    st.caption("Tell us who you are so the assistant can personalize alerts.")

    name = st.text_input("Name", value=st.session_state.profile.get("name", ""))
    age = st.number_input("Age", min_value=1, max_value=120, value=int(st.session_state.profile.get("age", 25)))
    language = st.selectbox("Preferred language", options=list(LANGUAGE_CODES.keys()), index=list(LANGUAGE_CODES.keys()).index(st.session_state.profile.get("preferred_language", "English")))

    health_conditions = st.multiselect(
        "Health conditions",
        options=["Diabetes", "Hypertension", "Pregnancy", "Kidney disease", "Heart condition", "Liver condition"],
        default=st.session_state.profile.get("health_conditions", []),
    )

    allergies = st.multiselect(
        "Allergy keywords",
        options=["milk", "egg", "peanut", "tree nut", "soy", "wheat", "gluten", "fish", "shellfish", "sesame"],
        default=st.session_state.profile.get("allergies", []),
    )

    if st.button("Continue to Scan", type="primary", use_container_width=True):
        if not name.strip():
            st.warning("Please enter your name.")
        else:
            st.session_state.profile = {
                "name": name.strip(),
                "age": int(age),
                "preferred_language": language,
                "health_conditions": health_conditions,
                "allergies": [a.lower() for a in allergies],
            }
            st.session_state.step = 2
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_step_2(api_base: str) -> None:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Step 2 - Scan")

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
            selected_mime = "image/jpeg"
            st.image(selected_bytes, caption="Captured image", use_container_width=True)
    else:
        uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])
        if uploaded is not None:
            selected_name = uploaded.name
            selected_bytes = uploaded.getvalue()
            selected_mime = uploaded.type or "image/jpeg"
            st.image(selected_bytes, caption="Uploaded image", use_container_width=True)

    if st.button("Start Scan", type="primary", use_container_width=True):
        if not selected_bytes:
            st.warning("Please upload or capture an image first.")
        else:
            st.session_state.scan_image_name = selected_name
            st.session_state.scan_image_bytes = selected_bytes
            st.session_state.scan_image_mime = selected_mime
            with st.spinner("Scanning your food label..."):
                try:
                    payload = api_analyze_image(api_base, selected_name, selected_bytes, selected_mime)
                except RequestException:
                    st.error("Unable to process. Please try again.")
                    return

            ingredients = payload.get("ai_ingredients") or payload.get("ingredients", [])
            ingredients = clean_ingredients_for_ui([str(x) for x in ingredients])

            st.session_state.ingredients = ingredients
            st.session_state.display_ingredients = ingredients
            st.session_state.ingredient_editor = ", ".join(ingredients)
            st.session_state.ocr_source = str(payload.get("source", "fallback"))
            st.session_state.allergens = [str(x).lower() for x in payload.get("allergens", []) if str(x).strip()]
            st.session_state.prediction = None
            st.session_state.prediction_signature = ""
            st.session_state.step = 3
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_step_3() -> None:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Step 3 - Extracted Ingredients")
    st.caption("Showing cleaned ingredients from the scan. You can edit before continuing.")

    render_ingredient_chips(st.session_state.ingredients)

    st.text_area(
        "Editable ingredient list",
        key="ingredient_editor",
        height=140,
        placeholder="milk, sugar, cocoa butter",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Scan", type="secondary", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("Continue to Clean & Translate", type="primary", use_container_width=True):
            edited = clean_ingredients_for_ui(parse_ingredient_text(st.session_state.ingredient_editor))
            if not edited:
                st.warning("Please provide at least one valid ingredient.")
            else:
                st.session_state.ingredients = edited
                st.session_state.display_ingredients = edited
                st.session_state.processing_signature = ""
                st.session_state.step = 4
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_step_4() -> None:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Step 4 - Clean & Translate")

    preferred_language = st.session_state.profile.get("preferred_language", "English")
    current_signature = ingredient_signature(st.session_state.ingredients, preferred_language)

    if st.session_state.processing_signature != current_signature:
        with st.spinner("Optimizing ingredient list..."):
            cleaned = clean_ingredients_for_ui(st.session_state.ingredients)
            translated, note = translate_with_libretranslate(cleaned, preferred_language)
            st.session_state.ingredients = cleaned
            st.session_state.display_ingredients = translated
            st.session_state.translation_note = note
            st.session_state.processing_signature = current_signature

    st.markdown("<div class='section-label'>Cleaned Ingredients</div>", unsafe_allow_html=True)
    render_ingredient_chips(st.session_state.ingredients)

    if preferred_language.lower() != "english":
        st.markdown("<div class='section-label'>Translated Ingredients</div>", unsafe_allow_html=True)
        render_ingredient_chips(st.session_state.display_ingredients)

    if st.session_state.translation_note:
        st.caption(st.session_state.translation_note)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Ingredients", type="secondary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("Continue to Analyze", type="primary", use_container_width=True):
            st.session_state.step = 5
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_step_5() -> None:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Step 5 - Analyze")
    st.caption("Analyzing food safety using our hybrid ML model")

    options = ["Good", "Average", "Poor", "Not sure"]

    st.session_state.hygiene_choice = st.radio(
        "Hygiene (VHI)",
        options=options,
        horizontal=True,
        index=options.index(st.session_state.hygiene_choice),
    )
    st.session_state.awareness_choice = st.radio(
        "Awareness (CAS)",
        options=options,
        horizontal=True,
        index=options.index(st.session_state.awareness_choice),
    )
    st.session_state.environment_choice = st.radio(
        "Environment (ERF)",
        options=options,
        horizontal=True,
        index=options.index(st.session_state.environment_choice),
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Clean & Translate", type="secondary", use_container_width=True):
            st.session_state.step = 4
            st.rerun()
    with col2:
        if st.button("Run Analysis", type="primary", use_container_width=True):
            st.session_state.step = 6
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_results_sections(result: Dict, ingredients: List[str], translated_ingredients: List[str]) -> None:
    risk_class = str(result.get("risk_classification", "Unknown"))
    risk_score = float(result.get("risk_score", result.get("probability", 0.0))) * 100.0

    if risk_class.lower() == "low":
        risk_color = "var(--ok)"
    elif risk_class.lower() == "medium":
        risk_color = "var(--warn)"
    else:
        risk_color = "var(--danger)"

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Step 6 - Results")

    st.markdown("<div class='section-label'>Model Prediction</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='risk-box'>
            <div style='opacity:.85;'>Risk Card</div>
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

    st.markdown("<div class='section-label'>Why this risk? (Model output)</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='risk-box'>
            <div>Ingredient risk: <strong>{ingredient_risk:.2f}</strong></div>
            <div>ERF contribution: <strong>{erf_value:.2f}</strong></div>
            <div>Allergen count: <strong>{len(allergens_detected)}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-label'>AI Explanation</div>", unsafe_allow_html=True)
    ai_expl = str(result.get("ai_explanation", "")).strip()
    fallback_expl = str(result.get("explanation", "")).strip()
    explanation_text = ai_expl or fallback_expl or "No AI explanation available for this run."
    st.markdown("<div class='risk-box'><strong>AI-generated explanation</strong><br/>" + explanation_text + "</div>", unsafe_allow_html=True)

    recommendations = [str(x).strip() for x in result.get("recommendations", []) if str(x).strip()]
    if recommendations:
        st.markdown("<div class='risk-box'><strong>AI recommendations</strong>", unsafe_allow_html=True)
        for rec in recommendations[:3]:
            st.markdown(f"- {rec}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Ingredients Breakdown</div>", unsafe_allow_html=True)
    if st.session_state.profile.get("preferred_language", "English").lower() != "english":
        st.caption("Translated for display")
        render_ingredient_chips(translated_ingredients)
        st.caption("Canonical cleaned ingredients")
        render_ingredient_chips(ingredients)
    else:
        render_ingredient_chips(ingredients)

    alert = st.session_state.personal_alert
    st.markdown("<div class='section-label'>Personal Alert</div>", unsafe_allow_html=True)
    if alert:
        st.error(alert)
    else:
        st.success("No profile-specific allergen conflict detected.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_how_it_works() -> None:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("How this works")
    st.markdown(
        """
        <div class='how-card'>
            <div><strong>1) OCR -> extraction</strong>: The backend reads the food label image and extracts ingredient text.</div>
            <div><strong>2) Cleaning -> preprocessing</strong>: Rule-based logic removes OCR noise and normalizes ingredients.</div>
            <div><strong>3) ML model -> prediction</strong>: A hybrid model uses VHI, CAS, ERF, and ingredient risk features.</div>
            <div><strong>4) AI -> explanation</strong>: AI text is used only to explain and recommend, not to make predictions.</div>
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
        with st.spinner("Running safety analysis..."):
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

    render_results_sections(st.session_state.prediction, ingredients, translated_ingredients)
    render_how_it_works()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Analyze", type="secondary", use_container_width=True):
            st.session_state.step = 5
            st.rerun()
    with col2:
        if st.button("Analyze Another Label", type="primary", use_container_width=True):
            st.session_state.step = 2
            st.session_state.ingredients = []
            st.session_state.display_ingredients = []
            st.session_state.ingredient_editor = ""
            st.session_state.prediction = None
            st.session_state.prediction_signature = ""
            st.session_state.translation_note = ""
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="EatWise AI", page_icon="EW", layout="wide")

    init_state()
    apply_theme()

    api_base = get_api_base()

    st.markdown(
        """
        <div class='hero'>
            <h1 style='margin:0;font-family:Manrope,sans-serif;'>EatWise AI Assistant</h1>
            <p style='margin:.35rem 0 0;color:var(--muted);'>A guided step-by-step workflow for food safety assessment with clear ML and AI separation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_progress(st.session_state.step)

    step = st.session_state.step
    if step == 1:
        render_step_1()
    elif step == 2:
        render_step_2(api_base)
    elif step == 3:
        render_step_3()
    elif step == 4:
        render_step_4()
    elif step == 5:
        render_step_5()
    else:
        render_step_6(api_base)


if __name__ == "__main__":
    main()
