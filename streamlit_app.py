import streamlit as st
import pandas as pd
import joblib

st.write("## Stroke Prediction using Machine Learning")

st.write("""
Predicting whether a patient is likely to get a stroke or not
""")

st.image("images/cover.jpeg")

st.write("""
## About

<p align="justify">Stroke is the 2nd leading cause of death globally, responsible for approximately 11% of
total deaths (WHO). Stroke is a medical condition characterized by disrupted blood supply to the brain,
leading to cellular death. Signs and symptoms of a stroke may include an inability to move or feel on one side
of the body, problems understanding or speaking, dizziness, or loss of vision to one side. 
<br />
This project aim to predict whether a patient is likely to get stroke based on input parameters
like gender, age, various diseases, smoking status and so on. Everything you need to know regarding this
project including the documentation, notebook, dataset, evaluation metric, models etc. can be found in my repository on <a href="https://github.com/Oyebamiji-Micheal/Stroke-Prediction-Web-App" target="_blank" style="text-decoration: None">Github</a>.</p>
""", unsafe_allow_html=True)

st.write("""
**Made by Oyebamiji Micheal**
""")

predict_stroke_status = st.button("Predict Stroke Likelihood")

st.sidebar.header("User Input Features")

age = st.sidebar.slider("Age", min_value=1, max_value=100)

gender = st.sidebar.selectbox("Gender", ("Male", "Female"))

avg_glucose_level = st.sidebar.number_input("Enter Average Glucose Level", step=0.01, min_value=40.0, max_value=300.0)

hypertension = st.sidebar.number_input(
    "Hypertension: Enter 1 if patient has hypertension and 0 otherwise", min_value=0, max_value=1
)

heart_disease = st.sidebar.number_input(
    "Heart Disease: Enter 1 if patient has an heart disease and 0 otherwise", min_value=0, max_value=1
)

ever_married = st.sidebar.selectbox("Ever Married?", ("Yes", "No"))

residence_type = st.sidebar.selectbox("Residence Type", ("Urban", "Rural"))

smoking_status = st.sidebar.selectbox(
    "Smoking Status", ("Never Smoked", "Formerly Smoked", "Smokes", "Unknown")
)

work_type = st.sidebar.selectbox(
    "Work Type", ("Private", "Self-employed", "Government job", "Children", "Never Worked")
)

bmi = st.sidebar.number_input("BMI: Enter patient BMI ", min_value=10, max_value=100)


def predict_input(single_input):
    stroke_model = joblib.load("stroke_model.joblib")
    input_df = pd.DataFrame([single_input])

    encoded_cols, numeric_cols = stroke_model["encoded_cols"], stroke_model["numeric_cols"]
    preprocessor = stroke_model["preprocessor"]    
    input_df[encoded_cols] = preprocessor.transform(input_df)

    X = input_df[numeric_cols + encoded_cols]
    prediction = stroke_model['model'].predict(X)

    return prediction


if predict_stroke_status:
    # Format inputs
    gender = gender.lower()
    ever_married = ever_married.lower()
    smoking_status = smoking_status.lower() if smoking_status != "Unknown" else smoking_status
    work_type_mapping = {
        "Government job": "Govt_job", "Children": "children", 
        "Never Worked": "Never_worked", "Private": "Private"
    }
    
    single_input = {
        "gender":  gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type_mapping[work_type],
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status,
    }

    prediction = predict_input(single_input)

    st.write(f"Classifier = Logistic Regression with SMOTE")

    if prediction[0] == 1:
        st.write("Predicted Stroke Status = Likely")
    else:
        st.write("Predicted Status = Not Likely")
