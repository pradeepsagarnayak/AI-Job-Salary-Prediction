import streamlit as st
import pandas as pd
import joblib

model = joblib.load("salary_pipeline.pkl")

st.title("AI Job Salary Prediction")

# Extra Inputs (only for UI)
job_title = st.text_input("Job Title")
experience_level = st.selectbox("Experience Level", ["EN", "MI", "SE", "EX"])
employment_type = st.selectbox("Employment Type", ["FT", "PT", "CT", "FL"])
company_location = st.text_input("Company Location")
company_size = st.selectbox("Company Size", ["S", "M", "L"])
employee_residence = st.text_input("Employee Residence")
required_skills = st.text_input("Required Skills")
education_required = st.text_input("Education Required")
industry = st.text_input("Industry")
company_name = st.text_input("Company Name")

# Features actually used by the model
years_experience = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=50,
    value=1
)

benefits_score = st.number_input(
    "Benefits Score",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)

remote_ratio = st.selectbox(
    "Remote Ratio",
    [0, 50, 100]
)

if st.button("Predict Salary"):

    # Only these 3 columns go to the model
    input_data = pd.DataFrame({
        "years_experience": [years_experience],
        "benefits_score": [benefits_score],
        "remote_ratio": [remote_ratio]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Salary: ${prediction[0]:,.2f}")
