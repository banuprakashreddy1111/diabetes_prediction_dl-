import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os

# -------------------------------------------------------------
# ADVANCED UI DESIGN (CSS)
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# AUTO-TRAIN MODEL (Only first time)
# -------------------------------------------------------------
if not os.path.exists("diabetes_dl_model.keras"):
    st.write("⏳ Training deep learning model for the first time...")

    # Load dataset (place your CSV in GitHub repo)
    df = pd.read_csv("diabetes.csv")   # <-- your dataset

    X = df.drop("Diabetes", axis=1)
    y = df["Diabetes"]

    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    # Save model + scaler
    model.save("diabetes_dl_model.keras")
    pickle.dump(scaler, open("scaler.pkl", "wb"))

    st.success("🎉 Model trained and saved successfully!")

# -------------------------------------------------------------
# LOAD MODEL & SCALER
# -------------------------------------------------------------
model = tf.keras.models.load_model("diabetes_dl_model.keras", compile=False)
scaler = pickle.load(open("scaler.pkl", "rb"))

# -------------------------------------------------------------
# FEATURE LIST
# -------------------------------------------------------------
feature_list = [
    "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke",
    "HeartDiseaseorAttack","PhysActivity","Fruits","Veggies",
    "HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth",
    "MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"
]

ui_labels = {
    "HighBP": "High Blood Pressure",
    "HighChol": "High Cholesterol",
    "CholCheck": "Cholesterol Check Recently",
    "BMI": "Body Mass Index",
    "Smoker": "Smoker",
    "Stroke": "Stroke History",
    "HeartDiseaseorAttack": "Heart Disease/Attack",
    "PhysActivity": "Physical Activity",
    "Fruits": "Fruit Consumption",
    "Veggies": "Vegetable Consumption",
    "HvyAlcoholConsump": "Heavy Alcohol Use",
    "AnyHealthcare": "Has Health Insurance",
    "NoDocbcCost": "Avoided Doctor Visit (Cost)",
    "GenHlth": "General Health (1–5)",
    "MentHlth": "Bad Mental Health Days (0–30)",
    "PhysHlth": "Bad Physical Health Days (0–30)",
    "DiffWalk": "Difficulty Walking",
    "Sex": "Gender (1=Male, 0=Female)",
    "Age": "Age Group Category",
    "Education": "Education Level",
    "Income": "Income Level"
}

# -------------------------------------------------------------
# ADVANCED UI
# -------------------------------------------------------------
st.markdown('<div class="main-title">🧬 Diabetes Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter your details below</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
inputs = []

for i, feature in enumerate(feature_list):
    label = ui_labels[feature]

    if i < len(feature_list)/2:
        with col1:
            val = st.number_input(f"**{label}**", min_value=0.0, step=1.0)
    else:
        with col2:
            val = st.number_input(f"**{label}**", min_value=0.0, step=1.0)

    inputs.append(val)

# -------------------------------------------------------------
# PREDICTION
# -------------------------------------------------------------
if st.button("🔍 Predict Diabetes", use_container_width=True):
    x = np.array([inputs])
    x_scaled = scaler.transform(x)
    prob = model.predict(x_scaled)[0][0]

    st.write("### Prediction Result")

    if prob > 0.5:
        st.markdown(f"""
            <div class="result-box error-box">
                🔴 Diabetes Risk Detected  
                <br>Probability: {prob:.4f}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-box success-box">
                🟢 No Diabetes Risk  
                <br>Probability: {prob:.4f}
            </div>
        """, unsafe_allow_html=True)

    st.progress(float(prob))




