import streamlit as st
import tensorflow as tf
import pickle
import numpy as np

# Load model and scaler
model = tf.keras.models.load_model("diabetes_dl_model.h5")
scaler = pickle.load(open("scaler.pkl","rb"))

# Correct feature list (No duplicates)
feature_list = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
    'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
    'Income'
]

st.title("🧬 Diabetes Prediction App")
st.write("Enter patient details below and click **Predict** 👇")

# Collect inputs with unique keys
user_inputs = []
for i, feature in enumerate(feature_list):
    val = st.number_input(
        f"Enter {feature}",
        value=0.0,
        key=f"num_{feature}_{i}"   # ✅ unique key fix
    )
    user_inputs.append(val)

# Prediction
if st.button("Predict Diabetes"):
    inp = np.array([user_inputs])
    inp = scaler.transform(inp)
    pred = model.predict(inp)[0][0]

    if pred > 0.5:
        st.error("🔴 **Diabetes Risk Detected!**")
    else:
        st.success("🟢 **No Diabetes**")

    st.write("Probability score:", round(pred, 4))
