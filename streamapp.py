import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('logistic_regression_model.pkl')

# Title of the app
st.title("Lung Cancer Prediction App")

# Step 1: User inputs for the features
st.header("Input the health details")

# Example input features (replace with relevant features from your dataset)
gender = st.selectbox("Gender", options=['Male', 'Female'])
age = st.slider("Age", 21, 87, 50)
smoking = st.selectbox("Smoking", options=[0, 1])
yellow_fingers = st.selectbox("Yellow Fingers", options=[0, 1])
anxiety = st.selectbox("Anxiety", options=[0, 1])
peer_pressure = st.selectbox("Peer Pressure", options=[0, 1])
chronic_disease = st.selectbox("Chronic Disease", options=[0, 1])
fatigue = st.selectbox("Fatigue", options=[0, 1])
allergy = st.selectbox("Allergy", options=[0, 1])
wheezing = st.selectbox("Wheezing", options=[0, 1])
alcohol = st.selectbox("Alcohol Consuming", options=[0, 1])
coughing = st.selectbox("Coughing", options=[0, 1])
shortness_of_breath = st.selectbox("Shortness of Breath", options=[0, 1])
swallowing_difficulty = st.selectbox("Swallowing Difficulty", options=[0, 1])
chest_pain = st.selectbox("Chest Pain", options=[0, 1])

# Step 2: Convert user inputs into a format the model can use
gender_numeric = 1 if gender == 'Male' else 0  # Assuming 'Male' is 1 and 'Female' is 0

# Arrange the inputs in the correct order as per the model's training
input_features = np.array([[age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease,
                            fatigue, allergy, wheezing, alcohol, coughing, shortness_of_breath,
                            swallowing_difficulty, chest_pain, gender_numeric]])

# Step 3: Predict lung cancer based on input
if st.button("Predict"):
    prediction = model.predict(input_features)
    
    # Step 4: Show the result
    if prediction[0] == 1:
        st.subheader("The model predicts: High Risk of Lung Cancer")
    else:
        st.subheader("The model predicts: Low Risk of Lung Cancer")
