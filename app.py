import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# LOAD TRAINED OBJECTS (replace with your actual saved files)
# ---------------------------------------------------------
import joblib
scaler = joblib.load("scaler.pkl")
svm_model = joblib.load("svm_model.pkl")
feature_order = joblib.load("feature_order.pkl")  # list of engineered features in correct order

st.title("🎓 Student Risk Prediction (SVM Model)")
st.write("Enter the student's information below. The model will compute engineered features and generate a prediction.")


# =========================================================
# INPUT FIELDS
# =========================================================

st.header("📘 Study Information")

study_weekdays = st.number_input("Study Hours on Weekdays", min_value=0.0, step=0.5)
study_weekends = st.number_input("Study Hours on Weekends", min_value=0.0, step=0.5)

late_submissions = st.selectbox(
    "Late Submission Frequency",
    options=[1, 2, 3, 4],
    format_func=lambda x: {1: "Never", 2: "Rarely", 3: "Sometimes", 4: "Often"}[x]
)

academic_units = st.number_input("Number of Academic Units", min_value=1.0, step=1.0)


st.header("🎮 Gaming & Time Use")

gaming_play = st.radio("Do you play games?", ["No", "Yes"])
gaming_hours = 0
if gaming_play == "Yes":
    gaming_hours = st.number_input("Gaming Hours per Day", min_value=0.0, step=0.5)

part_time = st.radio("Do you work part-time?", ["No", "Yes"])
work_hours = 0
if part_time == "Yes":
    work_hours = st.number_input("Work Hours per Week", min_value=0.0, step=1.0)


st.header("🏫 Extracurricular")

extra_involved = st.radio("Are you involved in extracurricular activities?", ["No", "Yes"])
extracurricular_hours = 0
if extra_involved == "Yes":
    extracurricular_hours = st.number_input("Extracurricular Hours per Week", min_value=0.0, step=1.0)


st.header("🧠 Well-Being Information")

stress_level = st.slider("Stress Level (1-5)", 1, 5, 3)
social_support = st.slider("Social Support (1-5)", 1, 5, 3)
sleep_hours = st.number_input("Sleep Hours per Night", min_value=0.0, step=0.5)
financial_difficulty = st.slider("Financial Difficulty (1-5)", 1, 5, 3)


# =========================================================
# FEATURE ENGINEERING
# =========================================================
if st.button("🔍 Predict Risk"):

    total_study = study_weekdays + study_weekends

    # Compute engineered features exactly like your script
    user_data = {
        "Total Study Hours": total_study,
        "StudyEfficiency": total_study / (late_submissions + 0.1),
        "AcademicEngagement": extracurricular_hours + social_support,
        "StressBalance": stress_level - social_support,
        "TimeBurden": work_hours + gaming_hours,
        "StudyGamingRatio": total_study / (gaming_hours if gaming_hours > 0 else 0.1),
        "SleepStudyRatio": sleep_hours / (total_study + 1),
        "StudyPerUnit": total_study / (academic_units + 0.1),
    }

    # Convert to DataFrame
    user_df = pd.DataFrame([user_data])

    # Ensure every feature exists in correct order
    for col in feature_order:
        if col not in user_df.columns:
            user_df[col] = 0

    user_df = user_df[feature_order]

    # Scale input
    scaled_input = scaler.transform(user_df)

    # Predict with SVM
    pred = svm_model.predict(scaled_input)[0]
    proba = svm_model.predict_proba(scaled_input)[0]

    prob_at_risk = proba[1]
    prob_not_risk = proba[0]

    # Display results
    st.subheader("📊 Prediction Result")
    if pred == 1:
        st.error(f"🚨 **AT-RISK** (Confidence: {prob_at_risk:.1%})")
    else:
        st.success(f"✅ **NOT AT-RISK** (Confidence: {prob_not_risk:.1%})")

    st.write("### Probability Breakdown")
    st.write(f"- **At-Risk:** {prob_at_risk:.1%}")
    st.write(f"- **Not At-Risk:** {prob_not_risk:.1%}")

    st.write("### Engineered Features Used")
    st.json(user_data)
