import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =================================================================
# IMPROVED SVG ICON DEFINITIONS (Better looking)
# =================================================================

SVG_ICONS = {
    "book": """
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
    </svg>
    """,
    
    "game": """
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="6 9 6 2 18 2 18 9"></polyline>
        <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path>
        <rect x="6" y="14" width="12" height="8"></rect>
    </svg>
    """,
    
    "work": """
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect>
        <path d="M16 3h-2a2 2 0 0 0-2 2v2H8V5a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v2"></path>
    </svg>
    """,
    
    "activities": """
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <polyline points="12 6 12 12 16 14"></polyline>
    </svg>
    """,
    
    "health": """
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
    </svg>
    """,
    
    "result_at_risk": """
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="12" y1="8" x2="12" y2="12"></line>
        <line x1="12" y1="16" x2="12.01" y2="16"></line>
    </svg>
    """,
    
    "result_safe": """
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="20 6 9 17 4 12"></polyline>
    </svg>
    """,
}

def render_svg(svg_name, color="currentColor"):
    """Render SVG icon with custom color"""
    if svg_name in SVG_ICONS:
        svg_code = SVG_ICONS[svg_name].replace("currentColor", color)
        st.markdown(
            f'<div style="display: inline-block; vertical-align: middle;">{svg_code}</div>',
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
        st.error(f"Error: Model files not found: {e}")
        st.info("Please ensure scaler.pkl, svm_model.pkl, and feature_order.pkl are in the app directory")
        return None, None, None

# =================================================================
# PAGE CONFIGURATION
# =================================================================

st.set_page_config(
    page_title="Student Risk Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("Student At-Risk Prediction System")
st.markdown("**Powered by Machine Learning (SVM Model)**")
st.write("Enter student information below to predict academic risk status.")

# =================================================================
# LOAD MODELS
# =================================================================

scaler, svm_model, feature_order = load_models()

if scaler is None or svm_model is None or feature_order is None:
    st.stop()

# =================================================================
# INPUT SECTIONS WITH IMPROVED ICONS
# =================================================================

st.markdown("---")

# Study Information Section
col1, col2 = st.columns([0.5, 20])
with col1:
    render_svg("book", "#2563eb")
with col2:
    st.subheader("Study Information")

col_a, col_b = st.columns(2)
with col_a:
    study_weekdays = st.number_input(
        "Study Hours on Weekdays",
        min_value=0.0,
        max_value=24.0,
        step=0.5,
        value=5.0,
        help="Average hours spent studying on weekdays"
    )

with col_b:
    study_weekends = st.number_input(
        "Study Hours on Weekends",
        min_value=0.0,
        max_value=24.0,
        step=0.5,
        value=3.0,
        help="Average hours spent studying on weekends"
    )

col_c, col_d = st.columns(2)
with col_c:
    late_submissions = st.selectbox(
        "Late Submission Frequency",
        options=["never", "rarely", "sometimes", "often"],
        help="How often do you submit assignments late?"
    )
    late_map = {"never": 1, "rarely": 2, "sometimes": 3, "often": 4}
    late_submissions_numeric = late_map[late_submissions]

with col_d:
    academic_units = st.number_input(
        "Academic Units (Last Semester)",
        min_value=1.0,
        max_value=50.0,
        step=1.0,
        value=15.0,
        help="Total academic units enrolled"
    )

st.markdown("---")

# Gaming & Time Use Section
col1, col2 = st.columns([0.5, 20])
with col1:
    render_svg("game", "#f59e0b")
with col2:
    st.subheader("Gaming & Time Use")

col_a, col_b = st.columns(2)
with col_a:
    gaming_play = st.radio(
        "Do you play games?",
        ["No", "Yes"],
        horizontal=True,
        key="gaming_play"
    )
    gaming_hours = 0.0
    if gaming_play == "Yes":
        gaming_hours = st.number_input(
            "Gaming Hours per Day",
            min_value=0.0,
            max_value=24.0,
            step=0.5,
            value=2.0,
            key="gaming_hours"
        )

with col_b:
    part_time = st.radio(
        "Do you work part-time?",
        ["No", "Yes"],
        horizontal=True,
        key="part_time"
    )
    work_hours = 0.0
    if part_time == "Yes":
        work_hours = st.number_input(
            "Work Hours per Week",
            min_value=0.0,
            max_value=168.0,
            step=1.0,
            value=10.0,
            key="work_hours"
        )

st.markdown("---")

# Extracurricular Section
col1, col2 = st.columns([0.5, 20])
with col1:
    render_svg("activities", "#8b5cf6")
with col2:
    st.subheader("Extracurricular Activities")

extra_involved = st.radio(
    "Involved in extracurricular activities?",
    ["No", "Yes"],
    horizontal=True,
    key="extra_involved"
)

extracurricular_hours = 0.0
if extra_involved == "Yes":
    extracurricular_hours = st.number_input(
        "Extracurricular Hours per Week",
        min_value=0.0,
        max_value=168.0,
        step=1.0,
        value=5.0,
        key="extracurricular_hours"
    )

st.markdown("---")

# Well-Being Information Section
col1, col2 = st.columns([0.5, 20])
with col1:
    render_svg("health", "#ef4444")
with col2:
    st.subheader("Well-Being Information")

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
        step=0.5,
        value=7.0
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
    
    col_icon, col_result = st.columns([0.8, 5])
    
    with col_icon:
        if pred == 1:
            render_svg("result_at_risk")
        else:
            render_svg("result_safe")
    
    with col_result:
        if pred == 1:
            st.error(
                f"### ⚠️ STUDENT IS AT-RISK\n"
                f"**Risk Confidence: {prob_at_risk:.1%}**",
                icon="⚠️"
            )
            st.warning(
                "This student shows indicators of academic risk. "
                "Consider providing academic support or intervention."
            )
        else:
            st.success(
                f"### ✅ STUDENT IS NOT AT-RISK\n"
                f"**Safety Confidence: {prob_not_risk:.1%}**",
                icon="✅"
            )
            st.info(
                "This student appears to be on track academically."
            )
    
    # Probability Breakdown
    st.markdown("---")
    st.subheader("Prediction Details")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "At-Risk Probability",
            f"{prob_at_risk:.1%}"
        )
    with col2:
        st.metric(
            "Not At-Risk Probability",
            f"{prob_not_risk:.1%}"
        )
    with col3:
        st.metric(
            "Prediction",
            "🚨 AT-RISK" if pred == 1 else "✅ SAFE"
        )
    
    # Probability bar chart
    prob_data = pd.DataFrame({
        "Status": ["At-Risk", "Not At-Risk"],
        "Probability": [prob_at_risk, prob_not_risk]
    })
    
    st.bar_chart(prob_data.set_index("Status"), height=300)
    
    # Engineered Features Display
    st.markdown("---")
    st.subheader("Engineered Features Analysis")
    
    features_df = pd.DataFrame(
        list(user_data.items()),
        columns=["Feature", "Value"]
    )
    features_df["Value"] = features_df["Value"].round(3)
    
    st.dataframe(features_df, use_container_width=True, hide_index=True)
    
    # Feature Explanation
    with st.expander("📖 What do these features mean?"):
        st.markdown("""
        - **Total Study Hours**: Weekday + weekend study combined
        - **StudyEfficiency**: Study relative to late submission rate
        - **AcademicEngagement**: Extracurricular involvement + social support
        - **StressBalance**: Stress vs. emotional support difference
        - **TimeBurden**: Work + gaming hours combined
        - **StudyGamingRatio**: Balance between study and leisure
        - **SleepStudyRatio**: Rest quality relative to study intensity
        - **StudyPerUnit**: Study effort per course enrolled
        """)

st.markdown("---")
st.caption("🤖 Powered by SVM Model | 98% Test Accuracy")
