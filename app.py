import streamlit as st
import joblib
import numpy as np
import time

# ========================
# Load Models and Scaler
# ========================
scaler = joblib.load("models/final_scaler.pkl")
cov_model = joblib.load("models/best_coverage_model.pkl")
pre_model = joblib.load("models/best_premium_model.pkl")

# Manual mappings (same as used in training)
gender_map = {"Male": 1, "Female": 0}
smoker_map = {"Yes": 1, "No": 0}
occupation_map = {
    "IT": 0,
    "Finance": 1,
    "Healthcare": 2,
    "Manufacturing": 3,
    "Education": 4,
    "Retail": 5,
    "Government": 6,
    "Other": 7
}

# ========================
# Preprocessing Function
# ========================
def preprocess_input(age, gender, annual_income, dependents, smoker_status, occupation_type, bmi):
    gender_val = gender_map.get(gender, 0)
    smoker_val = smoker_map.get(smoker_status, 0)
    occupation_val = occupation_map.get(occupation_type, 7)

    log_income = np.log1p(annual_income)
    coverage_to_income = 10
    premium_to_income = 0.05

    age_group = 0 if age < 30 else (1 if age <= 50 else 2)
    bmi_cat = 0 if bmi < 18.5 else (1 if bmi <= 24.9 else (2 if bmi <= 29.9 else 3))

    high_risk = 1 if (smoker_val == 1 or bmi > 30) else 0
    dependents_risk = 1 if dependents > 3 else 0

    features = np.array([[
        age, gender_val, annual_income, dependents, smoker_val,
        occupation_val, bmi, coverage_to_income, premium_to_income,
        age_group, bmi_cat, high_risk, dependents_risk, log_income
    ]])
    return scaler.transform(features)

# ========================
# Streamlit App Layout
# ========================
st.set_page_config(page_title="Insurance Predictor", page_icon="💼", layout="wide")

# Title
st.markdown(
    """
    <h1 style='text-align: center; color: #1E90FF;'>💼 Insurance Coverage & Premium Predictor</h1>
    <p style='text-align: center; color: #555;'>
        Enter your details to get a personalized estimate for your health insurance coverage and premium.
    </p>
    """, unsafe_allow_html=True
)

# Two-column form layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    age = st.number_input("Age (Years)", min_value=18, max_value=100, value=30, step=1)
    gender = st.selectbox("Gender", list(gender_map.keys()))
    annual_income = st.number_input("Annual Income (₹)", min_value=100000, max_value=10000000, value=500000, step=10000)
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1, step=1)

with col2:
    st.subheader("Lifestyle & Job")
    smoker_status = st.radio("Smoker Status", list(smoker_map.keys()), horizontal=True)
    occupation_type = st.selectbox("Occupation Type", list(occupation_map.keys()))
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=24.5, step=0.1)

st.markdown("---")

# Predict Button
if st.button("🔍 Predict My Insurance Plan", use_container_width=True):
    with st.spinner("Analyzing your profile..."):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)
        progress.empty()

    # Preprocess and Predict
    X_input = preprocess_input(age, gender, annual_income, dependents, smoker_status, occupation_type, bmi)
    cov_pred = cov_model.predict(X_input)[0]
    pre_pred = pre_model.predict(X_input)[0]

    # Results Card
    st.markdown(
        f"""
        <div style='background-color:#f0f8ff;padding:20px;border-radius:15px;margin-top:20px;'>
            <h2 style='color:#1E90FF;'>Your Insurance Plan</h2>
            <p><b>Suggested Health Coverage:</b> <span style='color:green;font-size:22px;'>₹{cov_pred:,.0f}</span></p>
            <p><b>Estimated Monthly Premium:</b> <span style='color:orange;font-size:22px;'>₹{pre_pred:,.0f}</span></p>
        </div>
        """, unsafe_allow_html=True
    )

    # Risk Warning or Info
    if smoker_status == "Yes" or bmi > 30:
        st.warning("⚠️ You fall into a higher risk category (Smoker or BMI > 30). Premiums may be higher.")
    else:
        st.info("✅ You have a standard risk profile. Premiums are in the typical range.")

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: #888;'>
        Built with ❤️ using Machine Learning and Streamlit
    </p>
    """, unsafe_allow_html=True
)
