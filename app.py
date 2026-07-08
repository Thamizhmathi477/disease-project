import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="🏥 AI Disease Diagnosis", layout="wide")

st.title("🏥 AI Disease Diagnosis System")

# ---- Load all files with joblib (correct) ----
try:
    with st.spinner("Loading model..."):
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        encoder = joblib.load('encoder.pkl')
        features = joblib.load('features.pkl')
        diseases = joblib.load('diseases.pkl')
    st.success(f"✅ Model loaded! {len(diseases)} diseases, {len(features)} symptoms")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ---- Sidebar Navigation ----
with st.sidebar:
    st.markdown("### 🏥 Navigation")
    page = st.radio("", ["🏠 Home", "🩺 Disease Prediction", "📄 About"])

# ---- Home ----
if page == "🏠 Home":
    st.markdown("""
    Welcome to the **AI-powered Disease Diagnosis System**.
    - Go to **Disease Prediction** to check your symptoms.
    - Get instant diagnosis with confidence score.
    - Download your report.
    """)

# ---- Disease Prediction ----
elif page == "🩺 Disease Prediction":
    st.write("Select all symptoms that apply:")

    selected = []
    cols = st.columns(4)
    for i, sym in enumerate(features):
        col = cols[i % 4]
        if col.checkbox(sym.replace('_', ' ').title(), key=sym):
            selected.append(sym)

    if st.button("🔍 Predict", type="primary"):
        if not selected:
            st.warning("Please select at least one symptom.")
        else:
            with st.spinner("Analyzing..."):
                vec = np.zeros(len(features))
                for sym in selected:
                    if sym in features:
                        vec[features.index(sym)] = 1
                scaled = scaler.transform([vec])
                pred = model.predict(scaled)[0]
                disease = diseases[pred]
                probs = model.predict_proba(scaled)[0]
                confidence = max(probs) * 100

                st.markdown("---")
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"""
                    <div style="background:#E3F2FD; padding:20px; border-radius:10px; border-left:6px solid #1976D2;">
                        <h3>🦠 Diagnosis</h3>
                        <h2 style="color:#0D47A1;">{disease}</h2>
                        <p>Confidence: <strong>{confidence:.2f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if confidence > 80:
                        st.error("🔴 High Risk - Consult a doctor.")
                    elif confidence > 60:
                        st.warning("🟡 Medium Risk - Monitor symptoms.")
                    else:
                        st.success("🟢 Low Risk - Take rest.")

                report = f"""
=====================================
AI DIAGNOSIS REPORT
=====================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Disease: {disease}
Confidence: {confidence:.2f}%
Symptoms: {', '.join(selected)}
=====================================
"""
                st.download_button("📄 Download Report", report, file_name=f"diagnosis_{datetime.now().strftime('%Y%m%d')}.txt", mime="text/plain")

# ---- About ----
else:
    st.markdown("""
    **AI-Based Disease Diagnosis System**  
    Built with Streamlit and Random Forest.  
    This is an educational project.  
    **Always consult a healthcare professional.**
    """)
