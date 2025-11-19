"""
Streamlit User Interface for Credit Risk Analysis System
"""
import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any, List
from credit_form_interface import (
    create_credit_application_form,
    create_credit_application_form_m,
    custom_labels,
    field_options,
)

st.set_page_config(
    page_title="Credit Risk Analysis",
    page_icon="💳",
    layout="wide",  # 👈 garantiza que ocupe todo el ancho
    initial_sidebar_state="expanded"
)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")  + "/api/v1"

def login_ui(page_prefix):
    # --- Diseño centrado ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style="
                background-color: #f8f9fa;
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;">
                <h2 style="color:#1f77b4;">🔐 Credit Risk Analysis Login</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        email = st.text_input("👤 Email", placeholder="example@domain.com", key=f"{page_prefix}_email")
        password = st.text_input("🔑 Password", type="password", key=f"{page_prefix}_password")

        if st.button("Login", use_container_width=True):
            if not email or not password:
                st.warning("⚠️ Please enter both email and password.")
                st.stop()

            try:
                response = requests.post(
                    f"{API_BASE_URL}/auth/login",
                    json={"email": email, "password": password},
                    timeout=5
                )
                if response.status_code == 200:
                    token = response.json().get("access_token")
                    role = response.json().get("role")
                    st.session_state["token"] = token
                    st.session_state["role"] = role
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")
            except Exception as e:
                st.error(f"🚨 Error connecting to authentication server: {e}")

def signup_ui(page_prefix):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="
            background-color: #f8f9fa;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;">
            <h2 style="color:#1f77b4;">🧾 Create New Account</h2>
        </div>
        """, unsafe_allow_html=True)

        email = st.text_input("📧 Email", key=f"{page_prefix}_email")
        full_name = st.text_input("🧍 Full Name", key=f"{page_prefix}_fullname")
        password = st.text_input("🔑 Password", type="password", key=f"{page_prefix}_password")
        confirm_password = st.text_input("🔁 Confirm Password", type="password", key=f"{page_prefix}_confirm")

        if st.button("Create Account", use_container_width=True):
            # --- Validaciones básicas ---
            if not email or not full_name or not password or not confirm_password:
                st.warning("⚠️ Please fill in all fields.")
                st.stop()
            if "@" not in email:
                st.warning("⚠️ Please enter a valid email address.")
                st.stop()
            if password != confirm_password:
                st.error("❌ Passwords do not match.")
                st.stop()

            try:
                response = requests.post(
                    f"{API_BASE_URL}/auth/signup",
                    json={
                        "email": email,
                        "full_name": full_name,
                        "password": password
                    },
                    timeout=5
                )
                if response.status_code in (200, 201):
                    st.success("✅ Account created successfully! Please log in.")
                    st.info("You can now return to the Login page.")
                else:
                    st.error(f"❌ Could not create account. Server says: {response.text}")
            except Exception as e:
                    st.error(f"🚨 Error connecting to backend: {e}")

# ---------------------------------
# NAVEGACIÓN Y CONTROL DE SESIÓN
# ---------------------------------
if "token" not in st.session_state:
    # Mostrar solo login / signup mientras no haya sesión
    page = st.sidebar.radio("Navigation", ["Login", "Sign up"])

    if page == "Login":
        login_ui(page)
    elif page == "Sign up":
        signup_ui(page)
    st.stop()  # 👈 Detiene aquí, no ejecuta el resto del código


# ---------------------------------
# STYLES    
# ---------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 10px;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# HELPER FUNCTIONS
# ---------------------------------
def check_api_health() -> bool:
    """Check API availability."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_single_profile(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Send individual prediction to API."""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=profile_data, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def predict_batch_profiles(profiles_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Send batch predictions to API."""
    try:
        response = requests.post(f"{API_BASE_URL}/predict/batch", json={"profiles": profiles_data}, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Batch request failed: {e}")
        return None

def simulate_credit_decisions(profiles_data: List[Dict[str, Any]], decision_threshold: float, profit_margin: float):
    """Simulate credit decision profitability."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/simulate",
            json={"profiles": profiles_data, "decision_threshold": decision_threshold, "profit_margin": profit_margin},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        return None

def build_model_payload_from_form(form: Dict[str, Any]) -> Dict[str, Any]:
    """Mapea el formulario completo a los 5 campos requeridos por el modelo.

    - income = PERSONAL_MONTHLY_INCOME + OTHER_INCOMES
    - age = AGE
    - employment_length = floor(MONTHS_IN_THE_JOB / 12)
    - credit_amount ≈ 20% de PERSONAL_ASSETS_VALUE (si falta, 10000)
    - debt_ratio ≈ credit_amount / (income*12 + assets), recortado a [0, 0.9]
    """
    def to_float(x, default=0.0):
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    def to_int(x, default=0):
        try:
            if x is None:
                return default
            return int(float(x))
        except Exception:
            return default

    income = to_float(form.get("PERSONAL_MONTHLY_INCOME", 0)) + to_float(form.get("OTHER_INCOMES", 0))
    age = to_int(form.get("AGE", 30), 30)
    months_job = to_int(form.get("MONTHS_IN_THE_JOB", 0), 0)
    employment_length = max(0, months_job // 12)
    assets = to_float(form.get("PERSONAL_ASSETS_VALUE", 0))
    credit_amount = max(1000.0, round(assets * 0.2, 2))

    denom = max(1.0, income * 12.0 + assets)
    debt_ratio = min(0.9, max(0.0, credit_amount / denom))

    return {
        "income": float(income),
        "age": int(age),
        "credit_amount": float(credit_amount),
        "employment_length": int(employment_length),
        "debt_ratio": float(debt_ratio),
    }

def display_risk_result(prediction: Dict[str, Any]):
    """Display risk prediction with correct color and layout."""
    # --- Extract values ---
    risk_score = float(prediction.get("risk_score", 0))
    confidence = prediction.get("confidence", 0)
    recommendation = prediction.get("recommendation", "Review")

    # --- Determine label and color ---
    if risk_score >= 0.7:
        risk_level = "BAD"
        color = "#f44336"  # Red
        decision = "🚫 Reject"
        explanation = "⚠️ High risk of default — profile should be rejected."
    elif 0.4 <= risk_score < 0.7:
        risk_level = "MEDIUM"
        color = "#ff9800"  # Orange
        decision = "🟠 Review"
        explanation = "⚠️ Medium risk — requires manual review."
    else:
        risk_level = "GOOD"
        color = "#4caf50"  # Green
        decision = "✅ Approve"
        explanation = "✅ Low risk — client likely to meet obligations."

    # --- Display metrics in same row ---

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background-color:{color}20;padding:10px;border-radius:8px;text-align:center;">
            <h4 style="margin:0;">Risk Level</h4>
            <p style="font-size:24px;font-weight:bold;color:{color};margin:0;">{risk_level}</p>
            <p style="color:{color};margin:0;">Score: {risk_score:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background-color:{color}20;padding:10px;border-radius:8px;text-align:center;">
            <h4 style="margin:0;">Decision</h4>
            <p style="font-size:24px;font-weight:bold;color:{color};margin:0;">{decision}</p>
            <p style="color:{color};margin:0;">Confidence: {confidence:.2f}</p>
        </div>
        """, unsafe_allow_html=True)


    # --- Gauge Visualization ---
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        title={'text': "Risk Score (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "#e8f5e8"},
                {'range': [40, 70], 'color': "#fff3e0"},
                {'range': [70, 100], 'color': "#ffebee"}
            ]
        }
    ))
    fig.update_layout(height=400)  # Increased size
    st.plotly_chart(fig, use_container_width=True)

    # --- Summary Card ---
    st.markdown(f"""
    <div style="background-color:{color}20;padding:10px;border-radius:8px;">
        <strong>Interpretation:</strong> {explanation}<br>
        <strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------
# MAIN APP
# ---------------------------------
def main():
    with st.sidebar:
        st.markdown(f"👋 Logged in as: **{st.session_state.get('role', 'Unknown')}**")
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()
    st.markdown('<h1 class="main-header">💳 Credit Risk Analysis System</h1>', unsafe_allow_html=True)

    if not check_api_health():
        st.error("⚠️ API not available. Please run: `uvicorn api.main:app --reload`")
        st.stop()

    # Sidebar Navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🔍 Individual Analysis", "📊 Batch Analysis", "🎯 Decision Simulation", "📈 Metrics Dashboard"]
    )

    # --- INDIVIDUAL ANALYSIS ---
    if page == "🔍 Individual Analysis":
        st.subheader("Individual Credit Application")
        profile_data = create_credit_application_form_m(custom_labels, field_options)  
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔮 Predict Credit Risk", type="primary",use_container_width=True):
                rejection_flags = [
                        "FLAG_HOME_ADDRESS_DOCUMENT","FLAG_RG","FLAG_CPF","FLAG_INCOME_PROOF","FLAG_ACSP_RECORD"
                        ]

                # --- Identify bad flags (value N or 1) ---
                bad_flags = [
                    f for f in rejection_flags
                    if str(profile_data.get(f, "")).strip().upper() in ["N", "1"]
                ]

                # Si hay FLAGs en N → perfil malo (score = 0)
                if bad_flags:
                    st.error("🚫 Credit Profile: **Bad (Rejected)**")
                    st.write("The following fields caused rejection:")
                    for f in bad_flags:
                        st.write(f"•", f)
                    risk_score = 1.0
                    recommendation = "Reject Application"
                else:
                    # Caso normal: consultar modelo
                    with st.spinner("Analyzing credit profile..."):
                        model_payload = build_model_payload_from_form(profile_data)
                        result = predict_single_profile(model_payload)
                    if result:
                        risk_score = result.get("risk_score", 0.5)  # valor entre 0 y 1
                        recommendation = result.get("recommendation", "Review")
                    else:
                        st.error("Failed to get prediction.")
                        risk_score = None

                
            # Mostrar resultado con display_risk_result
                if risk_score is not None:
                    result = {
                        "risk_score": risk_score,
                        "confidence": 1.0,
                        "recommendation": recommendation
                    }
                    display_risk_result(result)




    # --- BATCH ANALYSIS ---
    elif page == "📊 Batch Analysis":
        st.subheader("Batch Credit Profile Upload")
        profile_data = create_credit_application_form()

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔮 Predict Credit Risk", type="primary",use_container_width=True):
                yn_fields = [k for k in profile_data.keys() if k.startswith("FLAG_")]
                bad_flags = [f for f in yn_fields if str(profile_data.get(f, "")).upper() == "N"]

                # Si hay FLAGs en N → perfil malo (score = 0)
                if bad_flags:
                    st.error("🚫 Credit Profile: **Bad (Rejected)**")
                    st.write("The following fields caused rejection:")
                    for f in bad_flags:
                        st.write(f"•", f)
                    risk_score = 1.0
                    recommendation = "Reject Application"
                else:
                    # Caso normal: consultar modelo
                    with st.spinner("Analyzing credit profile..."):
                        model_payload = build_model_payload_from_form(profile_data)
                        result = predict_single_profile(model_payload)
                    if result:
                        risk_score = result.get("risk_score", 0.5)  # valor entre 0 y 1
                        recommendation = result.get("recommendation", "Review")
                    else:
                        st.error("Failed to get prediction.")
                        risk_score = None

                
            # Mostrar resultado con display_risk_result
                if risk_score is not None:
                    result = {
                        "risk_score": risk_score,
                        "confidence": 1.0,
                        "recommendation": recommendation
                    }
                    display_risk_result(result)


    # --- DECISION SIMULATION ---
    elif page == "🎯 Decision Simulation":
        st.subheader("Credit Decision Simulation")
        decision_threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)
        profit_margin = st.slider("Profit Margin", 0.0, 0.5, 0.05, 0.01)

        if st.button("🚀 Run Simulation"):
            import numpy as np
            np.random.seed(42)
            simulated_profiles = [
                {
                    "income": float(np.random.normal(50000, 15000)),
                    "age": int(np.random.randint(18, 80)),
                    "credit_amount": float(np.random.uniform(1000, 50000)),
                    "employment_length": int(np.random.randint(0, 30)),
                    "debt_ratio": float(np.random.uniform(0, 1))
                } for _ in range(300)
            ]
            with st.spinner("Running simulation..."):
                result = simulate_credit_decisions(simulated_profiles, decision_threshold, profit_margin)
            if result:
                st.success("✅ Simulation completed.")
                st.json(result)

    # --- METRICS DASHBOARD ---
    elif page == "📈 Metrics Dashboard":
        st.subheader("System and Model Metrics")
        try:
            response = requests.get(f"{API_BASE_URL}/model/info")
            if response.status_code == 200:
                model_info = response.json()
                st.json(model_info)
            else:
                st.warning("Unable to fetch model info.")
        except:
            st.warning("API not responding.")

if __name__ == "__main__":
    main()
