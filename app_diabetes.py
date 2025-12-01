import streamlit as st
import tensorflow as tf
import pickle
import numpy as np

# Load the trained model and scaler
model = tf.keras.models.load_model("diabetes_dl_model.h5")
scaler = pickle.load(open("scaler.pkl","rb"))

# List of features used for prediction
feature_list = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
    'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
    'Income'
]

# Clear UI labels with full forms
ui_labels = {
    "HighBP": "High Blood Pressure (1 = Yes, 0 = No)",
    "HighChol": "High Cholesterol (1 = Yes, 0 = No)",
    "CholCheck": "Cholesterol Check Done Recently (1 = Yes, 0 = No)",
    "BMI": "Body Mass Index (BMI) Value (Example: 24.5, 30.2, 36.4 etc.)",
    "Smoker": "Smoking Habit (1 = Smoker, 0 = Non-Smoker)",
    "Stroke": "History of Stroke (1 = Yes, 0 = No)",
    "HeartDiseaseorAttack": "Heart Disease or Heart Attack History (1 = Yes, 0 = No)",
    "PhysActivity": "Does Regular Exercise / Physical Activity (1 = Yes, 0 = No)",
    "Fruits": "Consumes Fruits Regularly (1 = Daily/Often, 0 = Rarely/Never)",
    "Veggies": "Consumes Vegetables Regularly (1 = Daily/Often, 0 = Rarely/Never)",
    "HvyAlcoholConsump": "Heavy Alcohol Consumption Habit (1 = Yes, 0 = No)",
    "AnyHealthcare": "Has Medical Insurance / Healthcare Access (1 = Yes, 0 = No)",
    "NoDocbcCost": "Avoided Doctor Visit Due to Cost (1 = Yes, 0 = No)",
    "GenHlth": "General Health Rating (1 = Excellent → 5 = Very Poor)",
    "MentHlth": "Mental Health Affected Days in last 30 days (0-30)",
    "PhysHlth": "Physical Health Affected Days in last 30 days (0-30)",
    "DiffWalk": "Difficulty in Walking or Climbing Stairs (1 = Yes, 0 = No)",
    "Sex": "Gender (1 = Male, 0 = Female)",
    "Age": "Age Group Category (Higher number = older age group)",
    "Education": "Education Level (1 = Low → 6 = Highest Education)",
    "Income": "Income Level Category (1 = Low → 8 = High Income)"
}

# UI
st.title("🧬 Diabetes Risk Prediction App")
st.markdown("### Fill patient details below ")

inputs = []
for i, feature in enumerate(feature_list):
    st.markdown(f"**{ui_labels.get(feature, feature)}**")
    v = st.number_input(" ", 0.0, key=f"{feature}_{i}")
    inputs.append(v)

# Predict button
if st.button("Predict Diabetes"):
    x = np.array([inputs])
    try:
        x = scaler.transform(x)
        p = model.predict(x)[0][0]

        if p > 0.5:
            st.error("🔴 **Diabetes Risk Detected!**")
        else:
            st.success("🟢 **No Diabetes Risk**")

        st.write("**Probability Score:**", round(p, 4))

    except Exception as e:
        st.write("⚠ Error during prediction:", e)
