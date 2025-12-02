import streamlit as st
import tensorflow as tf
import pickle
import numpy as np

# Load model and scaler
model = tf.keras.models.load_model("diabetes_dl_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

# Custom CSS for advanced UI
st.markdown("""
    <style>
        .main-title {
            font-size: 38px;
            font-weight: bold;
            text-align: center;
            color: #4bb3fd;
            text-shadow: 1px 1px 3px #00000030;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 20px;
            text-align: center;
            color: #ffffff;
            background-color: #4bb3fd;
            padding: 10px;
            border-radius: 10px;
        }
        .result-box {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
        }
        .success-box {
            background-color: #e3ffe6;
            color: #008a1e;
            border-left: 8px solid #008a1e;
        }
        .error-box {
            background-color: #ffe6e6;
            color: #b80000;
            border-left: 8px solid #b80000;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">🧬 Diabetes Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter your health details below to get prediction</div>', unsafe_allow_html=True)
st.write("")

# Feature list in order
feature_list = [
    "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack",
    "PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare",
    "NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex",
    "Age","Education","Income"
]

# UI Labels
ui_labels = {
    "HighBP": "High Blood Pressure",
    "HighChol": "High Cholesterol",
    "CholCheck": "Cholesterol Check Recently",
    "BMI": "Body Mass Index (BMI)",
    "Smoker": "Smoking Habit",
    "Stroke": "History of Stroke",
    "HeartDiseaseorAttack": "Heart Disease / Attack",
    "PhysActivity": "Physical Activity",
    "Fruits": "Fruit Consumption",
    "Veggies": "Vegetable Consumption",
    "HvyAlcoholConsump": "Heavy Alcohol Use",
    "AnyHealthcare": "Has Health Insurance",
    "NoDocbcCost": "Avoided Doctor Visit (Cost)",
    "GenHlth": "General Health Rating (1=Excellent → 5=Poor)",
    "MentHlth": "Bad Mental Health Days (0–30)",
    "PhysHlth": "Bad Physical Health Days (0–30)",
    "DiffWalk": "Difficulty Walking",
    "Sex": "Gender (1=Male, 0=Female)",
    "Age": "Age Group Category (1–13)",
    "Education": "Education Level (1–6)",
    "Income": "Income Level (1–8)"
}

# Two-column advanced layout
col1, col2 = st.columns(2)
inputs = []

for i, feature in enumerate(feature_list):
    label = ui_labels[feature]

    if i < len(feature_list)/2:
        with col1:
            val = st.number_input(f"**{label}**", min_value=0.0, step=1.0, key=f"{feature}_{i}")
    else:
        with col2:
            val = st.number_input(f"**{label}**", min_value=0.0, step=1.0, key=f"{feature}_{i}")

    inputs.append(val)

st.write("")
st.write("")

# Prediction button
if st.button("🔍 Predict Diabetes", use_container_width=True):
    x = np.array([inputs])
    x_scaled = scaler.transform(x)
    prob = model.predict(x_scaled)[0][0]

    st.write("### Prediction Result")
    
    if prob > 0.5:
        st.markdown(f"""
            <div class="result-box error-box">
                🔴 Diabetes Risk Detected  
                <br>Probability Score: {prob:.4f}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-box success-box">
                🟢 No Diabetes Risk  
                <br>Probability Score: {prob:.4f}
            </div>
        """, unsafe_allow_html=True)

    st.progress(float(prob))
