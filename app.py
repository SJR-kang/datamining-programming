import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =================================================================
# SVG ICON DEFINITIONS (Replace Emojis)
# =================================================================

SVG_ICONS = {
    "book": """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
        <path d="M5 3h14c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H5c-1.1 0-2-.9-2-2V5c0-1.1.9-2 2-2zm0 2v14h14V5H5z"/>
        <path d="M7 7h2v2H7zm4 0h2v2h-2zm4 0h2v2h-2zM7 11h2v2H7zm4 0h2v2h-2zm4 0h2v2h-2z"/>
    </svg>
    """,
    "game": """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
        <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z"/>
    </svg>
    """,
    "work": """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
        <path d="M20 6h-2.18c.11-.89.32-1.75.64-2.59.44-1.26.33-2.64-.3-3.84-.91-1.6-2.93-2.57-4.61-2.57-.82 0-1.61.24-2.29.67-.69-.43-1.47-.67-2.29-.67-1.68 0-3.7.97-4.61 2.57-.63 1.2-.74 2.58-.3 3.84.32.84.53 1.7.64 2.59H4c-1.1 0-2 .9-2 2v11c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm-5-2c.82 0 1.5.68 1.5 1.5S15.82 7 15 7c-.82 0-1.5-.68-1.5-1.5S14.18 4 15 4zm-6 0c.82 0 1.5.68 1.5 1.5S9.82 7 9 7c-.82 0-1.5-.68-1.5-1.5S8.18 4 9 4z"/>
    </svg>
    """,
    "activities": """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm3.5-9c.83 0 1.5-.67 1.5-1.5S16.33 8 15.5 8 14 8.67 14 9.5s.67 1.5 1.5 1.5zm-7 0c.83 0 1.5-.67 1.5-1.5S9.33 8 8.5 8 7 8.67 7 9.5 7.67 11 8.5 11zm3.5 6.5c2.33 0 4.31-1.46 5.11-3.5H6.89c.8 2.04 2.78 3.5 5.11 3.5z"/>
    </svg>
    """,
    "health": """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm3.5-9c.83 0 1.5.67 1.5 1.5s-.67 1.5-1.5 1.5-1.5-.67-1.5-1.5.67-1.5 1.5-1.5zm-7 0c.83 0 1.5.67 1.5 1.5S9.33 12 8.5 12 7 11.33 7 10.5 7.67 9 8.5 9zm3.5 6.5c2.33 0 4.31-1.46 5.11-3.5H6.89c.8 2.04 2.78 3.5 5.11 3.5z"/>
    </svg>
    """,
    "result_at_risk": """
    <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor">
        <circle cx="12" cy="12" r="10" fill="#dc2626"/>
        <path d="M12 7v5m0 4v1" stroke="white" stroke-width="2" stroke-linecap="round"/>
    </svg>
    """,
    "result_safe": """
    <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor">
        <circle cx="12" cy="12" r="10" fill="#16a34a"/>
        <path d="M10 13l2 2 4-4" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """
}

def render_svg(svg_name):
    """Render SVG icon in Streamlit"""
    if svg_name in SVG_ICONS:
        st.markdown(
            f'<div style="display:inline-block;">{SVG_ICONS[svg_name]}</div>',
            unsafe_allow_html=True
        )

# =================================================================
# LOAD TRAINED OBJECTS
# =================================================================

@st.cache_resource
def load_models():
    """Load pre-trained SVM model, scaler, and feature order"""
    try:
        scaler = joblib.load("scaler.pkl")
        svm_model = joblib.load("svm_model.pkl")
        feature_order = joblib.load("feature_order.pkl")
        return scaler, svm_model, feature_order
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.info("Please ensure you have saved: scaler.pkl, svm_model.pkl, feature_order.pkl")
        return None, None, None

# =================================================================
# PAGE CONFIGURATION
# =================================================================

st.set_page_config(
    page_title="Student Risk Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("🎓 Student At-Risk Prediction System")
st.markdown("**Using Support Vector Machine (SVM) Model**")
st.write("Enter the student's information below. The model will predict whether they are at risk of academic failure.")

# =================================================================
# LOAD MODELS
# =================================================================

scaler, svm_model, feature_order = load_models()

if scaler is None or svm_model is None or feature_order is None:
    st.stop()

# =================================================================
# INPUT SECTIONS
# =================================================================

st.markdown("---")

# Study Information Section
col1, col2 = st.columns([1, 20])
with col1:
    render_svg("book")
with col2:
    st.header("Study Information")

study_weekdays = st.number_input(
    "Study Hours on Weekdays",
    min_value=0.0,
    max_value=24.0,
    step=0.5,
    help="Average hours spent studying on weekdays"
)

study_weekends = st.number_input(
    "Study Hours on Weekends",
    min_value=0.0,
    max_value=24.0,
    step=0.5,
    help="Average hours spent studying on weekends"
)

late_submissions = st.selectbox(
    "Late Submission Frequency",
    options=["never", "rarely", "sometimes", "often"],
    help="How often do you submit assignments late?"
)

# Convert late submissions to numeric
late_map = {"never": 1, "rarely": 2, "sometimes": 3, "often": 4}
late_submissions_numeric = late_map[late_submissions]

academic_units = st.number_input(
    "Number of Academic Units (Last Semester)",
    min_value=1.0,
    max_value=50.0,
    step=1.0,
    help="Total academic units enrolled"
)

st.markdown("---")

# Gaming & Time Use Section
col1, col2 = st.columns([1, 20])
with col1:
    render_svg("game")
with col2:
    st.header("Gaming & Time Use")

gaming_play = st.radio(
    "Do you play games?",
    ["No", "Yes"],
    horizontal=True
)

gaming_hours = 0.0
if gaming_play == "Yes":
    gaming_hours = st.number_input(
        "Gaming Hours per Day",
        min_value=0.0,
        max_value=24.0,
        step=0.5
    )

part_time = st.radio(
    "Do you work part-time?",
    ["No", "Yes"],
    horizontal=True
)

work_hours = 0.0
if part_time == "Yes":
    work_hours = st.number_input(
        "Work Hours per Week",
        min_value=0.0,
        max_value=168.0,
        step=1.0
    )

st.markdown("---")

# Extracurricular Section
col1, col2 = st.columns([1, 20])
with col1:
    render_svg("activities")
with col2:
    st.header("Extracurricular Activities")

extra_involved = st.radio(
    "Are you involved in extracurricular activities?",
    ["No", "Yes"],
    horizontal=True
)

extracurricular_hours = 0.0
if extra_involved == "Yes":
    extracurricular_hours = st.number_input(
        "Extracurricular Hours per Week",
        min_value=0.0,
        max_value=168.0,
        step=1.0
    )

st.markdown("---")

# Well-Being Information Section
col1, col2 = st.columns([1, 20])
with col1:
    render_svg("health")
with col2:
    st.header("Well-Being Information")

col_stress, col_support = st.columns(2)
with col_stress:
    stress_level = st.slider(
        "Stress Level (1-5)",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Very Low, 5 = Very High"
    )

with col_support:
    social_support = st.slider(
        "Social Support (1-5)",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Very Low, 5 = Very High"
    )

col_sleep, col_financial = st.columns(2)
with col_sleep:
    sleep_hours = st.number_input(
        "Sleep Hours per Night",
        min_value=0.0,
        max_value=24.0,
        step=0.5
    )

with col_financial:
    financial_difficulty = st.slider(
        "Financial Difficulty (1-5)",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Very Low, 5 = Very High"
    )

# =================================================================
# PREDICTION BUTTON & RESULTS
# =================================================================

st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button(
        "🔮 Predict Risk Status",
        use_container_width=True,
        type="primary"
    )

if predict_button:
    # ========== FEATURE ENGINEERING ==========
    total_study = study_weekdays + study_weekends
    
    user_data = {
        "Total Study Hours": total_study,
        "StudyEfficiency": total_study / (late_submissions_numeric + 0.1),
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
    
    # ========== DISPLAY RESULTS ==========
    st.markdown("---")
    
    col_icon, col_result = st.columns([1, 4])
    
    with col_icon:
        if pred == 1:
            render_svg("result_at_risk")
        else:
            render_svg("result_safe")
    
    with col_result:
        if pred == 1:
            st.error(
                f"### ⚠️ STUDENT IS AT-RISK\n"
                f"**Confidence: {prob_at_risk:.1%}**",
                icon="⚠️"
            )
            st.warning(
                "This student shows indicators of being at-risk for academic failure. "
                "Consider providing academic support, counseling, or intervention programs."
            )
        else:
            st.success(
                f"### ✅ STUDENT IS NOT AT-RISK\n"
                f"**Confidence: {prob_not_risk:.1%}**",
                icon="✅"
            )
            st.info(
                "This student appears to be on track. Continue monitoring and provide "
                "support as needed."
            )
    
    # Probability Breakdown
    st.markdown("---")
    st.subheader("Prediction Probabilities")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "At-Risk Probability",
            f"{prob_at_risk:.1%}",
            delta=None
        )
    with col2:
        st.metric(
            "Not At-Risk Probability",
            f"{prob_not_risk:.1%}",
            delta=None
        )
    
    # Probability bar chart
    prob_data = pd.DataFrame({
        "Status": ["At-Risk", "Not At-Risk"],
        "Probability": [prob_at_risk, prob_not_risk]
    })
    
    st.bar_chart(prob_data.set_index("Status"))
    
    # Engineered Features Display
    st.markdown("---")
    st.subheader("Engineered Features Used for Prediction")
    
    features_df = pd.DataFrame(
        list(user_data.items()),
        columns=["Feature", "Value"]
    )
    features_df["Value"] = features_df["Value"].round(3)
    
    st.dataframe(features_df, use_container_width=True)
    
    # Feature Explanation
    with st.expander("📖 Understanding the Engineered Features"):
        st.markdown("""
        - **Total Study Hours**: Sum of weekday and weekend study hours
        - **StudyEfficiency**: Study hours relative to submission frequency (higher = more efficient)
        - **AcademicEngagement**: Combination of extracurricular involvement and social support
        - **StressBalance**: Difference between stress level and social support (lower = better balance)
        - **TimeBurden**: Combined work and gaming hours (higher = more time commitments)
        - **StudyGamingRatio**: Study hours compared to gaming (higher = more balanced)
        - **SleepStudyRatio**: Sleep hours relative to study hours (indicates rest quality)
        - **StudyPerUnit**: Study efficiency per academic unit enrolled
        """)

st.markdown("---")
st.caption("🤖 Model: Support Vector Machine (SVM) | Last Updated: 2025")
