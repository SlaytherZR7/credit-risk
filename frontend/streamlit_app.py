"""
Streamlit User Interface for Credit Risk Analysis System
"""
import os
import json
import numpy as np
import streamlit as st
import requests
import time
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
    layout="wide", 
    initial_sidebar_state="expanded"
)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")  + "/api/v1"

def login_ui(page_prefix):
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
if "token" not in st.session_state:

    page = st.sidebar.radio("Navigation", ["Login", "Sign up"])

    if page == "Login":
        login_ui(page)
    elif page == "Sign up":
        signup_ui(page)
    st.stop() 

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
def ensure_json_serializable(value):
    """Convert numpy, NaN, and other problematic types to JSON-safe Python types."""

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value

def check_api_health() -> bool:
    """Check API availability."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_single_profile(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Send individual prediction to API."""
    token = st.session_state.get("token") 

    if not token:
        st.error("❌ You must log in before making predictions.")
        return None

    headers = {
        "Authorization": f"Bearer {token}".strip(),
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predictions/predict-one",
            json={"features": [profile_data]},  
            headers=headers,     
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def predict_batch_profiles(profiles_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Send batch predictions to API with JSON cleaning + validation."""

    token = st.session_state.get("token")

    if not token:
        st.error("❌ You must log in before making predictions.")
        return None

    cleaned_profiles = []
    for profile in profiles_data:
        cleaned_profile = {k: ensure_json_serializable(v) for k, v in profile.items()}
        cleaned_profiles.append(cleaned_profile)

    st.write("📤 JSON FINAL enviado al backend:")
    st.json({"features": cleaned_profiles})

    try:
        json.dumps({"features": cleaned_profiles})
    except Exception as e:
        st.error(f"❌ INVALID JSON: {e}")
        return None


    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/predictions/predict-batch",
            json={"features": cleaned_profiles},
            headers=headers,
            timeout=30
        )

        st.write("📥 Backend response (raw):")
        st.write(response.status_code)
        st.write(response.text)

        response.raise_for_status()
        return response.json()

    except Exception as e:
        st.error(f"Batch request failed: {e}")
        return None


def build_model_payload_from_form(form_data: dict) -> dict:
    payload = {}

    for key, value in form_data.items():

        value = ensure_json_serializable(value)
        if value in ["", None, "None", "nan", "NaN"]:
            payload[key] = None

        elif isinstance(value, bool):
            payload[key] = int(value)

        elif isinstance(value, str) and value.isdigit():
            payload[key] = int(value)

        else:
            payload[key] = value

    return payload



def display_risk_result(prediction: Dict[str, Any]):
    """Display risk prediction with correct color and layout."""
    risk_score = float(prediction.get("risk_score", 0))
    confidence = prediction.get("confidence", 0)

    if risk_score >= 0.7:
        risk_level = "BAD"
        color = "#f44336" 
        decision = "🚫 Reject"
        explanation = "⚠️ High risk of default — profile should be rejected."
    elif 0.4 <= risk_score < 0.7:
        risk_level = "MEDIUM"
        color = "#ff9800" 
        decision = "🟠 Review"
        explanation = "⚠️ Medium risk — requires manual review."
    else:
        risk_level = "GOOD"
        color = "#4caf50"
        decision = "✅ Approve"
        explanation = "✅ Low risk — client likely to meet obligations."

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
    fig.update_layout(height=400) 
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div style="background-color:{color}20;padding:10px;border-radius:8px;">
        <strong>Interpretation:</strong> {explanation}<br>
        <strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

def fetch_job_result(job_id: str, max_attempts=20, sleep_time=1.0):
    """
    Poll the model service via API Gateway until job is finished.
    """
    token = st.session_state.get("token")
    if not token:
        st.error("❌ You must log in before fetching predictions.")
        return None
    headers = {
        "Authorization": f"Bearer {token}".strip(),
        "Content-Type": "application/json"
    }
    url = f"{API_BASE_URL}/predictions/result/{job_id}"
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            status = data.get("status")
            result = data.get("result")

            if status in ["finished", "failed"]:
                return data

        except Exception as e:
            st.error(f"Error fetching job result: {e}")
            return None

        time.sleep(sleep_time)

    st.error("⏳ Timeout waiting for batch result.")
    return None


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
    page = st.sidebar.radio(
        "Choose a section:",
        ["🔍 Individual Analysis", "📊 Batch Analysis"]
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
                bad_flags = []

                for f in rejection_flags:
                    value = str(profile_data.get(f, "")).strip().upper()

                    if f == "FLAG_ACSP_RECORD":

                        if value == "Y":
                            bad_flags.append(f)

                    else:
                        if value in ["N", "1"]:
                            bad_flags.append(f)

                if bad_flags:
                    st.error("🚫 Credit Profile: **Bad (Rejected)**")
                    st.write("The following fields caused rejection:")
                    for f in bad_flags:
                        st.write(f"•", f)
                    risk_score = 1.0
                    recommendation = "Reject Application"
                else:         
                    with st.spinner("Analyzing credit profile..."):
                        model_payload = build_model_payload_from_form(profile_data)
                        st.write(model_payload)
                        result = predict_single_profile(model_payload)
                    if result:
                        risk_score = result.get("risk_score", 0.5) 
                        recommendation = result.get("recommendation", "Review")
                    else:
                        st.error("Failed to get prediction.")
                        risk_score = None
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
                rejection_flags = [
                        "FLAG_HOME_ADDRESS_DOCUMENT","FLAG_RG","FLAG_CPF","FLAG_INCOME_PROOF","FLAG_ACSP_RECORD"
                        ]

                bad_flags = []

                for f in rejection_flags:
                    value = str(profile_data.get(f, "")).strip().upper()

                    if f == "FLAG_ACSP_RECORD":
                        if value == "Y":
                            bad_flags.append(f)

                    else:
                        if value in ["N", "1"]:
                            bad_flags.append(f)
                if bad_flags:
                    st.error("🚫 Credit Profile: **Bad (Rejected)**")
                    st.write("The following fields caused rejection:")
                    for f in bad_flags:
                        st.write(f"•", f)
                    risk_score = 1.0
                    recommendation = "Reject Application"
                else:
                    with st.spinner("Analyzing credit profile..."):
                        model_payload = build_model_payload_from_form(profile_data)
                        job_response  = predict_batch_profiles([model_payload]) 
                        st.write("📦 Response from batch enqueue:")
                        st.json(job_response)
                        st.subheader("📦 JSON enviado al modelo:")
                        st.code(json.dumps(model_payload, indent=2, ensure_ascii=False), language="json")
                    if job_response and "job_id" in job_response:
                        job_id = job_response["job_id"]
                        st.info(f"Job submitted with ID: {job_id}")

                    with st.spinner("Waiting for model prediction..."):
                        final_result = fetch_job_result(job_id)
                        st.write("📄 Final job response:")
                        st.json(final_result)

                    if final_result and final_result.get("status") == "finished":
                        output = final_result.get("result", {})
                        risk_score = output.get("risk_score")
                        recommendation = output.get("recommendation")
                    else:
                        st.error("Prediction failed or timed out.")
                        risk_score = None
             

                if risk_score is not None:
                    result = {
                        "risk_score": risk_score,
                        "confidence": 1.0,
                        "recommendation": recommendation
                    }
                    display_risk_result(result)

if __name__ == "__main__":
    main()
