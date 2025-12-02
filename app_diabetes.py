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
# Load model & scaler (unchanged)
# ---------------------------
# If you train inside the app previously, model file exists.
# Keep this as-is in your app.
MODEL_PATH = "diabetes_dl_model.keras"
SCALER_PATH = "scaler.pkl"

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scaler = pickle.load(open(SCALER_PATH, "rb"))
except Exception as e:
    st.error(f"Model or scaler not found or failed to load: {e}")
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

    /* page background */
    .stApp {
        background: linear-gradient(180deg, #f7fbff 0%, #f2fbff 40%, #ffffff 100%);
        color: #111827;
        font-family: "Inter", "Segoe UI", Roboto, Arial, sans-serif;
    }

    /* header */
    .hero {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        margin-bottom: 18px;
    }
    .hero h1{
        font-size: 48px;
        margin: 8px 0;
        color: var(--accent-1);
        text-shadow: 0 6px 20px rgba(47,128,237,0.12);
        letter-spacing: 0.6px;
    }
    .hero p {
        margin: 0;
        color: var(--muted);
        font-size: 16px;
    }

    /* container cards */
    .card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 30px rgba(31,41,55,0.06);
        border: 1px solid var(--glass-border);
        margin-bottom: 18px;
    }

    /* grid layout adjustments (works with st.columns) */
    .input-label {
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 6px;
    }

    /* big gradient button */
    .stButton>button {
        background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
        color: white;
        border: none;
        padding: 12px 18px;
        border-radius: 10px;
        font-size: 18px;
        box-shadow: 0 8px 30px rgba(47,128,237,0.12);
    }
    .stButton>button:hover { opacity: .95; transform: translateY(-1px); }

    /* big result circle */
    .result-circle {
        width: 180px;
        height: 180px;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 28px;
        color: white;
        margin: 12px auto;
        box-shadow: 0 10px 30px rgba(15,23,42,0.08);
    }

    .good-circle { background: linear-gradient(135deg, #34D399, #10B981); }
    .bad-circle { background: linear-gradient(135deg, #FF7A7A, #E11D48); }

    /* subtle small text */
    .muted { color: var(--muted); font-size: 13px; margin-top:6px; }

    /* sample profile chips */
    .chip {
        display:inline-block;
        padding:6px 10px;
        margin: 4px;
        border-radius:999px;
        background: linear-gradient(90deg, rgba(47,128,237,0.12), rgba(86,204,242,0.06));
        color: var(--accent-1);
        font-weight:600;
        cursor:pointer;
    }

    /* responsive tweaks */
    @media (max-width: 900px) {
        .hero h1{ font-size: 34px; }
        .result-circle { width: 140px; height:140px; font-size:22px; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Header / Hero
# ---------------------------
st.markdown(
    """
    <div class="hero">
      <div style="display:flex;align-items:center;gap:12px">
        <img src="https://raw.githubusercontent.com/edent/SuperTinyIcons/master/images/svg/dna.svg" width="48" />
        <h1>Diabetes Prediction System</h1>
      </div>
      <p>Advanced risk estimate powered by your trained model — modern UI, clear results.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Sidebar / quick controls
# ---------------------------
with st.sidebar:
    st.markdown("## 🔧 Quick controls")
    if st.button("Load sample: Healthy"):
        sample_choice = "healthy"
    if st.button("Load sample: At-risk"):
        sample_choice = "atrisk"
    st.markdown("---")
    st.markdown("**Model info**")
    try:
        st.write(f"Model: {MODEL_PATH}")
        st.write(f"Scaler: {SCALER_PATH}")
    except:
        pass
    st.markdown("---")
    st.caption("Tip: adjust inputs and press Predict")

# ---------------------------
# Feature definitions (min/max/step and nicer widgets)
# ---------------------------
# keep the original order expected by scaler/model
feature_defs = {
    "HighBP": dict(label="High Blood Pressure", kind="binary"),
    "HighChol": dict(label="High Cholesterol", kind="binary"),
    "CholCheck": dict(label="Cholesterol Check Done Recently", kind="binary"),
    "BMI": dict(label="Body Mass Index (BMI)", kind="float", min=10.0, max=60.0, step=0.1),
    "Smoker": dict(label="Smoker", kind="binary"),
    "Stroke": dict(label="History of Stroke", kind="binary"),
    "HeartDiseaseorAttack": dict(label="Heart Disease / Attack", kind="binary"),
    "PhysActivity": dict(label="Physical Activity (regular)", kind="binary"),
    "Fruits": dict(label="Fruits Consumption (daily)", kind="binary"),
    "Veggies": dict(label="Vegetables Consumption (daily)", kind="binary"),
    "HvyAlcoholConsump": dict(label="Heavy Alcohol Consumption", kind="binary"),
    "AnyHealthcare": dict(label="Has Health Insurance", kind="binary"),
    "NoDocbcCost": dict(label="Avoided Doctor Visit (cost)", kind="binary"),
    "GenHlth": dict(label="General Health (1=Excellent → 5=Poor)", kind="int", min=1, max=5, step=1),
    "MentHlth": dict(label="Bad Mental Health Days (0–30)", kind="int", min=0, max=30, step=1),
    "PhysHlth": dict(label="Bad Physical Health Days (0–30)", kind="int", min=0, max=30, step=1),
    "DiffWalk": dict(label="Difficulty Walking", kind="binary"),
    "Sex": dict(label="Gender (1=Male, 0=Female)", kind="binary"),
    "Age": dict(label="Age Group Category (1–13)", kind="int", min=1, max=13, step=1),
    "Education": dict(label="Education Level (1–6)", kind="int", min=1, max=6, step=1),
    "Income": dict(label="Income Level (1–8)", kind="int", min=1, max=8, step=1),
}

# default values for sample profiles
sample_profiles = {
    "healthy": {
        "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 23.5, "Smoker": 0, "Stroke": 0,
        "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 1, "MentHlth": 0, "PhysHlth": 0,
        "DiffWalk": 0, "Sex": 0, "Age": 4, "Education": 4, "Income": 5
    },
    "atrisk": {
        "HighBP": 1, "HighChol": 1, "CholCheck": 0, "BMI": 33.2, "Smoker": 1, "Stroke": 0,
        "HeartDiseaseorAttack": 1, "PhysActivity": 0, "Fruits": 0, "Veggies": 0, "HvyAlcoholConsump": 1,
        "AnyHealthcare": 0, "NoDocbcCost": 1, "GenHlth": 5, "MentHlth": 12, "PhysHlth": 10,
        "DiffWalk": 1, "Sex": 1, "Age": 10, "Education": 2, "Income": 2
    }
}

# ---------------------------
# Main input area (two-columns, glass cards)
# ---------------------------
left_col, right_col = st.columns([1, 1])

inputs = {}
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Patient Details", unsafe_allow_html=True)
    for key in list(feature_defs.keys())[:11]:
        fd = feature_defs[key]
        label = fd["label"]
        if fd["kind"] == "binary":
            # display as radio for clarity
            val = st.radio(label, options=[0,1], index=0, horizontal=True, key=f"r_{key}")
        elif fd["kind"] == "float":
            val = st.number_input(label, min_value=fd.get("min", 0.0), max_value=fd.get("max", 100.0), step=fd.get("step", 0.1), value=float(fd.get("min", 0.0)), key=f"n_{key}")
        else:
            val = st.number_input(label, min_value=int(fd.get("min",0)), max_value=int(fd.get("max",100)), step=int(fd.get("step",1)), value=int(fd.get("min",0)), key=f"n_{key}")
        inputs[key] = float(val)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Lifestyle & Health", unsafe_allow_html=True)
    for key in list(feature_defs.keys())[11:]:
        fd = feature_defs[key]
        label = fd["label"]
        if fd["kind"] == "binary":
            val = st.radio(label, options=[0,1], index=0, horizontal=True, key=f"r_{key}")
        elif fd["kind"] == "float":
            val = st.number_input(label, min_value=fd.get("min",0.0), max_value=fd.get("max",100.0), step=fd.get("step",0.1), value=float(fd.get("min",0.0)), key=f"n_{key}")
        else:
            val = st.number_input(label, min_value=int(fd.get("min",0)), max_value=int(fd.get("max",100)), step=int(fd.get("step",1)), value=int(fd.get("min",0)), key=f"n_{key}")
        inputs[key] = float(val)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# sample profile chips and controls
# ---------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Quick profiles")
col_a, col_b, col_c = st.columns([1,1,6])
with col_c:
    st.markdown("Choose a sample profile to auto-fill values.", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1,1,1])
if c1.button("Healthy sample"):
    for k,v in sample_profiles["healthy"].items():
        # set Streamlit widget state by rerunning with query params isn't trivial.
        # We'll simply show a preview and set inputs used for prediction.
        inputs.update(sample_profiles["healthy"])
    st.experimental_rerun()

if c2.button("At-risk sample"):
    inputs.update(sample_profiles["atrisk"])
    st.experimental_rerun()

if c3.button("Clear inputs"):
    for k in feature_defs.keys():
        inputs[k] = 0.0
    st.experimental_rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Prediction card
# ---------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Prediction", unsafe_allow_html=True)

# arrange action button and result in columns
act_col, out_col = st.columns([2,1])

with act_col:
    predict_clicked = st.button("🔍 Predict Diabetes Risk")

with out_col:
    placeholder = st.empty()
    # placeholder for result circle
    placeholder.markdown(
        """
        <div style="text-align:center; margin-top:6px;">
            <div style="width:180px; height:180px; border-radius:999px; background: linear-gradient(135deg,#F3F4F6,#ffffff); display:flex; align-items:center; justify-content:center; color:#111827;">
                <div style="font-size:14px; color:#6b7280">Probability</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------
# When user clicks predict
# ---------------------------
if predict_clicked:
    try:
        # ensure inputs in correct order used by model
        ordered_keys = [
            "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke",
            "HeartDiseaseorAttack","PhysActivity","Fruits","Veggies",
            "HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth",
            "MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"
        ]
        x = np.array([[inputs[k] for k in ordered_keys]], dtype=float)
        x_scaled = scaler.transform(x)
        prob = float(model.predict(x_scaled)[0][0])
        pct = int(round(prob*100))

        # result card + circle
        if prob > 0.5:
            color_class = "bad-circle"
            status_text = "High risk of Diabetes"
            status_sub = "Consult a healthcare provider"
        else:
            color_class = "good-circle"
            status_text = "Low risk of Diabetes"
            status_sub = "Maintain healthy habits"

        html = f"""
        <div style="text-align:center">
            <div class="{color_class} result-circle">{pct}%</div>
            <div style="font-weight:800; font-size:20px; text-align:center; margin-top:6px;">{status_text}</div>
            <div class="muted">{status_sub} — probability {prob:.4f}</div>
        </div>
        """
        placeholder.markdown(html, unsafe_allow_html=True)

        # progress bar
        st.progress(prob)

        # small details + actionable advice card
        st.markdown(
            f"""
            <div class="card">
                <b>Interpretation</b>
                <p class="muted">This score is a model-based probability. Use with clinical judgement.</p>
                <ul>
                  <li><b>Score:</b> {prob:.4f}</li>
                  <li><b>Status:</b> {status_text}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
st.markdown('<div style="text-align:center; color:#6b7280;">Built with ❤️ · Designed UI · Model predictions are illustrative</div>', unsafe_allow_html=True)



