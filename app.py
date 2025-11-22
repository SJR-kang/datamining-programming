import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Student Risk Prediction",
    page_icon="🎓",
    layout="centered"
)

# Title and description
st.title("🎓 Student Risk Prediction System")
st.markdown("---")

# Load or train model
@st.cache_resource
def load_model_and_scaler():
    try:
        # Try to load pre-trained model
        if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            return model, scaler
        else:
            # Train model if not exists
            st.info("🔄 Training model...")
            df = pd.read_csv('Balanced_Dataset_Expanded.csv')
            
            # Your preprocessing code here (same as before)
            df_clean = df.copy()
            # ... include all your preprocessing steps ...
            
            # For brevity, using the same preprocessing as before
            engineered_feature_names = ['Total Study Hours', 'StudyEfficiency', 'AcademicEngagement', 
                                      'StressBalance', 'TimeBurden', 'StudyGamingRatio', 
                                      'SleepStudyRatio', 'StudyPerUnit']
            
            X = df_clean[engineered_feature_names]
            y = df_clean['At-Risk/Not At-Risk']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
            
            lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
            lr.fit(X_train_res, y_train_res)
            
            # Save model
            with open('model.pkl', 'wb') as f:
                pickle.dump(lr, f)
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            
            return lr, scaler
            
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

# Reset form function
def reset_form():
    st.session_state.extracurricular_involved = "No"
    st.session_state.extracurricular_hours = 5.0
    st.session_state.gaming_question = "No"
    st.session_state.gaming_hours = 1.0
    st.session_state.part_time_work = "No"
    st.session_state.work_hours = 10.0
    st.session_state.study_weekdays = 4.0
    st.session_state.study_weekends = 2.0
    st.session_state.late_submissions = 1
    st.session_state.academic_units = 15
    st.session_state.stress_level = 3
    st.session_state.social_support = 3
    st.session_state.sleep_hours = 7.0
    st.session_state.financial_difficulty = 3

# Initialize session state for form inputs
if 'extracurricular_involved' not in st.session_state:
    reset_form()

# Load model
model, scaler = load_model_and_scaler()

if model and scaler:
    st.subheader("Enter the student's information:")
    
    # Reset button at the top
    col_buttons = st.columns([3, 1])
    with col_buttons[1]:
        if st.button("🔄 Reset All", use_container_width=True):
            reset_form()
            st.rerun()
    
    # --- OUTSIDE ACTIVITIES ---
    st.markdown("### 🎯 Outside Activities")
    
    # Extracurricular Activities
    extracurricular_involved = st.radio(
        "Are you involved in extracurricular activities?",
        ["No", "Yes"],
        horizontal=True,
        key="extracurricular_involved"
    )
    
    if extracurricular_involved == "Yes":
        extracurricular_hours = st.number_input(
            "Hours spent on extracurricular activities per week:",
            min_value=0.0,
            max_value=40.0,
            value=st.session_state.extracurricular_hours,
            step=0.5,
            key="extracurricular_hours"
        )
    
    # Gaming
    gaming_question = st.radio(
        "Are you playing games?",
        ["No", "Yes"],
        horizontal=True,
        key="gaming_question"
    )
    
    if gaming_question == "Yes":
        gaming_hours = st.number_input(
            "Hours spent playing games per day:",
            min_value=0.0,
            max_value=24.0,
            value=st.session_state.gaming_hours,
            step=0.5,
            key="gaming_hours"
        )
    
    # --- PART-TIME WORK ---
    st.markdown("### 💼 Part-time Work")
    
    part_time_work = st.radio(
        "Do you work part-time?",
        ["No", "Yes"],
        horizontal=True,
        key="part_time_work"
    )
    
    if part_time_work == "Yes":
        work_hours = st.number_input(
            "Work Hours per week:",
            min_value=0.0,
            max_value=40.0,
            value=st.session_state.work_hours,
            step=0.5,
            key="work_hours"
        )
    
    # --- STUDY INFORMATION ---
    st.markdown("### 📚 Study Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        study_weekdays = st.number_input(
            "Study Hours (Weekdays):",
            min_value=0.0,
            max_value=24.0,
            value=st.session_state.study_weekdays,
            step=0.5,
            key="study_weekdays"
        )
        
        study_weekends = st.number_input(
            "Study Hours (Weekends):",
            min_value=0.0,
            max_value=24.0,
            value=st.session_state.study_weekends,
            step=0.5,
            key="study_weekends"
        )
        
        late_submissions = st.selectbox(
            "Late Submissions frequency:",
            options=[1, 2, 3, 4],
            format_func=lambda x: ["Never", "Rarely", "Sometimes", "Often"][x-1],
            key="late_submissions"
        )
    
    with col2:
        academic_units = st.number_input(
            "Number of Academic Units:",
            min_value=0,
            max_value=30,
            value=st.session_state.academic_units,
            step=1,
            key="academic_units"
        )
    
    # --- WELL-BEING ---
    st.markdown("### 😊 Well-being")
    
    col3, col4 = st.columns(2)
    
    with col3:
        stress_level = st.slider(
            "Stress Level (1-5 scale):",
            min_value=1,
            max_value=5,
            value=st.session_state.stress_level,
            key="stress_level"
        )
        
        social_support = st.slider(
            "Level of Social Support (1-5 scale):",
            min_value=1,
            max_value=5,
            value=st.session_state.social_support,
            key="social_support"
        )
    
    with col4:
        sleep_hours = st.number_input(
            "Sleep Hours per night:",
            min_value=0.0,
            max_value=24.0,
            value=st.session_state.sleep_hours,
            step=0.5,
            key="sleep_hours"
        )
        
        financial_difficulty = st.slider(
            "Financial Difficulty (1-5 scale):",
            min_value=1,
            max_value=5,
            value=st.session_state.financial_difficulty,
            key="financial_difficulty"
        )
    
    # Predict button
    col_predict = st.columns([2, 1])
    with col_predict[0]:
        if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
            # Initialize variables for conditional inputs
            if extracurricular_involved == "No":
                extracurricular_hours = 0
            if gaming_question == "No":
                gaming_hours = 0
            if part_time_work == "No":
                work_hours = 0
            
            # Calculate engineered features
            total_study_hours = study_weekdays + study_weekends
            study_efficiency = total_study_hours / (late_submissions + 0.1)
            academic_engagement = extracurricular_hours + social_support
            stress_balance = stress_level - social_support
            time_burden = work_hours + gaming_hours
            study_gaming_ratio = total_study_hours / (gaming_hours if gaming_hours > 0 else 0.1)
            sleep_study_ratio = sleep_hours / (total_study_hours + 1)
            study_per_unit = total_study_hours / (academic_units if academic_units > 0 else 0.1)
            
            # Create feature array
            features = np.array([[
                total_study_hours, study_efficiency, academic_engagement,
                stress_balance, time_burden, study_gaming_ratio,
                sleep_study_ratio, study_per_unit
            ]])
            
            # Scale and predict
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            
            # Determine probabilities
            classes = model.classes_
            if 'At-Risk' in classes:
                at_risk_idx = list(classes).index('At-Risk')
                not_at_risk_idx = list(classes).index('Not At-Risk')
                prob_at_risk = probability[at_risk_idx]
                prob_not_risk = probability[not_at_risk_idx]
            else:
                prob_at_risk = probability[1] if prediction == 1 else probability[0]
                prob_not_risk = probability[0] if prediction == 1 else probability[1]
            
            # === RESULTS ===
            st.markdown("---")
            st.markdown("## 📊 PREDICTION RESULT")
            st.markdown("---")
            
            # Prediction and confidence
            col5, col6 = st.columns(2)
            
            with col5:
                if prediction == 'At-Risk' or prediction == 1:
                    st.error(f"### Prediction: **AT-RISK**")
                    confidence = prob_at_risk
                else:
                    st.success(f"### Prediction: **NOT AT-RISK**")
                    confidence = prob_not_risk
                
                st.metric("Confidence", f"{confidence:.1%}")
            
            with col6:
                st.subheader("Probability Breakdown")
                if 'At-Risk' in classes:
                    st.write(f"**Not At-Risk:** {probability[not_at_risk_idx]:.1%}")
                    st.write(f"**At-Risk:** {probability[at_risk_idx]:.1%}")
                else:
                    st.write(f"**Not At-Risk:** {probability[0]:.1%}")
                    st.write(f"**At-Risk:** {probability[1]:.1%}")
            
            # Key features display
            st.markdown("### 🔍 Key Calculated Features")
            
            col7, col8 = st.columns(2)
            
            with col7:
                st.write(f"**• Total Study Hours:** {total_study_hours:.1f}")
                st.write(f"**• Study Efficiency:** {study_efficiency:.1f}")
                st.write(f"**• Academic Engagement:** {academic_engagement:.1f}")
                st.write(f"**• Stress Balance:** {stress_balance:.1f}")
            
            with col8:
                st.write(f"**• Time Burden:** {time_burden:.1f}")
                st.write(f"**• Study-Gaming Ratio:** {study_gaming_ratio:.1f}")
                st.write(f"**• Financial Difficulty:** {financial_difficulty}")
                st.write(f"**• Part-time Work:** {'Yes' if part_time_work == 'Yes' else 'No'}")
                if part_time_work == "Yes":
                    st.write(f"**• Work Hours per week:** {work_hours}")

else:
    st.error("❌ Model not available. Please check your dataset.")

# Sidebar information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool predicts student academic risk using Logistic Regression 
    with 8 engineered features based on study habits and personal factors.
    
    **Features used:**
    - Total Study Hours
    - Study Efficiency  
    - Academic Engagement
    - Stress Balance
    - Time Burden
    - Study-Gaming Ratio
    - Sleep-Study Ratio
    - Study per Unit
    
    **Reset Button:** Click to clear all inputs and start over.
    """)
