import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import os

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Diabetes Risk — Premium",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------
# Load model & scaler
# ---------------------------
MODEL_PATH = "diabetes_dl_model.keras"
SCALER_PATH = "scaler.pkl"

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scaler = pickle.load(open(SCALER_PATH, "rb"))
except Exception as e:
    st.error(f"Model/scaler loading failed: {e}")
    st.stop()

# ---------------------------
# Premium CSS
# ---------------------------
st.markdown(
    """
    <style>
    :root{
        --accent-1: #2F80ED;
        --accent-2: #56CCF2;
        --card-bg: rgba(255,255,255,0.75);
        --glass-border: rgba(255,255,255,0.35);
        --muted: #6b7280;
    }

    .stApp {
        background: linear-gradient(180deg, #f7fbff 0%, #eefaff 40%, #ffffff 100%);
        font-family: "Inter", sans-serif;
    }

    .hero h1{
        font-size: 48px;
        color: var(--accent-1);
        text-shadow: 0 6px 18px rgba(47,128,237,0.15);
    }
    .hero{
        text-align:center;
        margin-bottom:15px;
    }

    .card{
        background: var(--card-bg);
        padding: 20px;
        border-radius: 16px;
        border: 1px solid var(--glass-border);
        box-shadow: 0 10px 28px rgba(0,0,0,0.07);
        margin-bottom: 18px;
    }

    .stButton>button {
        width: 100%;
        font-size: 18px;
        background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
        color: white;
        border: none;
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0 6px 20px rgba(47,128,237,0.25);
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        transition: 0.2s ease;
    }

    .result-circle {
        width: 170px;
        height: 170px;
        font-size: 26px;
        font-weight: bold;
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: auto;
    }
    .bad-circle {
        background: linear-gradient(130deg, #ff4d4d, #d10000);
        box-shadow: 0 6px 20px rgba(255,0,0,0.25);
    }
    .good-circle {
        background: linear-gradient(130deg, #14cc7a, #009e52);
        box-shadow: 0 6px 20px rgba(0,160,60,0.25);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------
# Header
# ---------------------------
st.markdown(
    """
    <div class='hero'>
        <h1>🧬 Diabetes Prediction System</h1>
        <p style="color:#6b7280; font-size:16px">Premium health risk estimation using machine learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Feature Definitions
# ---------------------------
feature_defs = {
    "HighBP": ("High Blood Pressure", "binary"),
    "HighChol": ("High Cholesterol", "binary"),
    "CholCheck": ("Cholesterol Check Recently", "binary"),
    "BMI": ("Body Mass Index", "float", 10.0, 60.0),
    "Smoker": ("Smoker", "binary"),
    "Stroke": ("Stroke History", "binary"),
    "HeartDiseaseorAttack": ("Heart Disease / Attack", "binary"),
    "PhysActivity": ("Physical Activity", "binary"),
    "Fruits": ("Fruit Consumption", "binary"),
    "Veggies": ("Vegetable Consumption", "binary"),
    "HvyAlcoholConsump": ("Heavy Alcohol Use", "binary"),
    "AnyHealthcare": ("Has Healthcare", "binary"),
    "NoDocbcCost": ("Avoided Doctor (Cost)", "binary"),
    "GenHlth": ("General Health (1=Best → 5=Worst)", "int", 1, 5),
    "MentHlth": ("Bad Mental Health Days (0–30)", "int", 0, 30),
    "PhysHlth": ("Bad Physical Health Days (0–30)", "int", 0, 30),
    "DiffWalk": ("Difficulty Walking", "binary"),
    "Sex": ("Gender (1=Male,0=Female)", "binary"),
    "Age": ("Age Category (1–13)", "int", 1, 13),
    "Education": ("Education Level (1–6)", "int", 1, 6),
    "Income": ("Income Level (1–8)", "int", 1, 8),
}

# For consistent order
ordered_keys = list(feature_defs.keys())

# ---------------------------
# Input Form (Premium Cards, 2 columns)
# ---------------------------
left, right = st.columns(2)
inputs = {}

with left:
    st.markdown("<div class='card'><h4>🩺 Patient Details</h4>", unsafe_allow_html=True)
    for key in ordered_keys[:11]:
        label, kind, *rest = feature_defs[key]

        if kind == "binary":
            inputs[key] = st.radio(label, [0,1], horizontal=True)
        elif kind == "float":
            minv, maxv = rest
            inputs[key] = st.slider(label, min_value=minv, max_value=maxv, step=0.1)
        else:
            minv, maxv = rest
            inputs[key] = st.slider(label, min_value=minv, max_value=maxv, step=1)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'><h4>🧑‍⚕️ Lifestyle & Health</h4>", unsafe_allow_html=True)
    for key in ordered_keys[11:]:
        label, kind, *rest = feature_defs[key]

        if kind == "binary":
            inputs[key] = st.radio(label, [0,1], horizontal=True)
        elif kind == "float":
            minv, maxv = rest
            inputs[key] = st.slider(label, min_value=minv, max_value=maxv, step=0.1)
        else:
            minv, maxv = rest
            inputs[key] = st.slider(label, min_value=minv, max_value=maxv, step=1)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# Prediction Button
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
colA, colB = st.columns([2,1])

with colA:
    predict_click = st.button("🔍 Predict Diabetes Risk")

with colB:
    result_box = st.empty()

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Prediction Logic
# ---------------------------
if predict_click:
    try:
        x = np.array([[inputs[k] for k in ordered_keys]])
        x_scaled = scaler.transform(x)
        prob = float(model.predict(x_scaled)[0][0])
        pct = int(prob * 100)

        if prob > 0.5:
            risk_html = f"""
            <div class='result-circle bad-circle'>
                {pct}%
            </div>
            <div style='text-align:center; font-size:20px; font-weight:700; color:#b80000'>
                High Diabetes Risk
            </div>
            <p style='text-align:center; color:#6b7280'>
                Probability Score: {prob:.4f}
            </p>
            """
        else:
            risk_html = f"""
            <div class='result-circle good-circle'>
                {pct}%
            </div>
            <div style='text-align:center; font-size:20px; font-weight:700; color:#008a1e'>
                Low Diabetes Risk
            </div>
            <p style='text-align:center; color:#6b7280'>
                Probability Score: {prob:.4f}
            </p>
            """

        result_box.markdown(risk_html, unsafe_allow_html=True)
        st.progress(prob)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown(
    """
    <div style='text-align:center; margin-top:18px; color:#6b7280'>
    Built with ❤️ | Premium UI | Machine Learning Diabetes Risk System
    </div>
    """,
    unsafe_allow_html=True,
)
