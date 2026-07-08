import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="🏥 AI Disease Diagnosis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS FOR BETTER VISIBILITY ----------
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #0D47A1 !important;
        font-weight: 600 !important;
    }
    
    /* Paragraph text */
    p, li, .stMarkdown {
        font-size: 16px !important;
        color: #1a1a2e !important;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-163i3l4 {
        background-color: #0D47A1 !important;
        color: white !important;
    }
    
    .css-1d391kg .stMarkdown, .css-163i3l4 .stMarkdown {
        color: white !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #1976D2 !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 10px 25px !important;
        border: none !important;
        width: 100% !important;
    }
    
    .stButton button:hover {
        background-color: #0D47A1 !important;
        box-shadow: 0 4px 15px rgba(13,71,161,0.3) !important;
    }
    
    /* Checkboxes */
    .stCheckbox label {
        font-size: 16px !important;
        color: #1a1a2e !important;
    }
    
    /* Diagnosis box */
    .diagnosis-box {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #1976D2;
        margin: 20px 0;
    }
    
    .diagnosis-box h2 {
        color: #0D47A1 !important;
        font-size: 32px !important;
    }
    
    .diagnosis-box h3 {
        color: #0D47A1 !important;
        font-size: 24px !important;
    }
    
    .diagnosis-box p {
        font-size: 18px !important;
        color: #1a1a2e !important;
    }
    
    /* Risk indicators */
    .risk-high {
        background: #FFEBEE;
        color: #D32F2F !important;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 18px;
        border-left: 4px solid #D32F2F;
    }
    
    .risk-medium {
        background: #FFF3E0;
        color: #F57C00 !important;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 18px;
        border-left: 4px solid #F57C00;
    }
    
    .risk-low {
        background: #E8F5E9;
        color: #388E3C !important;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 18px;
        border-left: 4px solid #388E3C;
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e8ecf0;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0D47A1;
    }
    
    .metric-label {
        color: #6c757d;
        font-size: 0.9rem;
        margin-top: 5px;
    }
    
    /* Symptom tags */
    .symptom-tag {
        background: #E3F2FD;
        color: #0D47A1;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 14px;
        margin: 3px;
        display: inline-block;
        border: 1px solid #BBDEFB;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 15px;
        border-top: 1px solid #e8ecf0;
        margin-top: 30px;
        font-size: 14px;
    }
    
    /* Download button */
    .stDownloadButton button {
        background-color: #43A047 !important;
        color: white !important;
    }
    
    .stDownloadButton button:hover {
        background-color: #2E7D32 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
st.title("🏥 AI Disease Diagnosis System")

try:
    with st.spinner("Loading model..."):
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        encoder = joblib.load('encoder.pkl')
        features = joblib.load('features.pkl')
        diseases = joblib.load('diseases.pkl')
    
    # Show metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(diseases)}</div>
            <div class="metric-label">🏥 Diseases</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(features)}</div>
            <div class="metric-label">🔬 Symptoms</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">88%</div>
            <div class="metric-label">📊 Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ---------- SIDEBAR NAVIGATION ----------
with st.sidebar:
    st.markdown("## 🏥 Navigation")
    page = st.radio("", ["🏠 Home", "🩺 Disease Prediction", "📄 About"])
    st.markdown("---")
    st.markdown(f"**Diseases:** {len(diseases)}")
    st.markdown(f"**Symptoms:** {len(features)}")

# ---------- HOME ----------
if page == "🏠 Home":
    st.markdown("""
    ### 👋 Welcome to the AI Disease Diagnosis System
    
    This system uses **Machine Learning** to predict diseases based on your symptoms.
    
    #### How it works:
    1. Go to **Disease Prediction**
    2. Select your symptoms
    3. Click **Predict**
    4. Get instant diagnosis with confidence score
    
    ⚠️ **Disclaimer:** This is an educational project. Always consult a healthcare professional.
    """)

# ---------- DISEASE PREDICTION ----------
elif page == "🩺 Disease Prediction":
    st.markdown("## 🩺 Disease Prediction")
    st.markdown("### Select all symptoms that apply:")

    selected = []
    cols = st.columns(4)
    for i, sym in enumerate(features):
        col = cols[i % 4]
        display_name = sym.replace('_', ' ').title()
        if col.checkbox(display_name, key=sym):
            selected.append(sym)

    if selected:
        st.markdown("#### 📋 Selected Symptoms:")
        tags = "".join([f'<span class="symptom-tag">✅ {s.replace("_", " ").title()}</span>' for s in selected])
        st.markdown(tags, unsafe_allow_html=True)

    if st.button("🔍 Predict Disease", type="primary"):
        if not selected:
            st.warning("⚠️ Please select at least one symptom.")
        else:
            with st.spinner("🧠 Analyzing symptoms..."):
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
                
                # Diagnosis result
                st.markdown("### 📊 Diagnosis Result")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"""
                    <div class="diagnosis-box">
                        <h3>🦠 Primary Diagnosis</h3>
                        <h2>{disease}</h2>
                        <p>Confidence: <strong>{confidence:.2f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if confidence > 80:
                        st.markdown('<div class="risk-high">🔴 High Risk - Consult a doctor immediately</div>', unsafe_allow_html=True)
                    elif confidence > 60:
                        st.markdown('<div class="risk-medium">🟡 Medium Risk - Monitor symptoms</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="risk-low">🟢 Low Risk - Take rest and stay hydrated</div>', unsafe_allow_html=True)
                
                # Emergency check
                emergency = ['chest_pain', 'shortness_breath', 'fainting', 'seizures']
                if any(s in emergency for s in selected):
                    st.error("🚨 **Emergency symptoms detected!** Seek immediate medical attention.")
                
                st.markdown("---")
                
                # Recommendations
                st.markdown("### 💊 Recommendations")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                    **💊 Medications**
                    - Consult a doctor
                    - Take prescribed medicines
                    - Do not self-medicate
                    """)
                with col2:
                    st.markdown("""
                    **🏠 Home Care**
                    - Get plenty of rest
                    - Stay hydrated
                    - Monitor symptoms
                    """)
                with col3:
                    st.markdown("""
                    **🥗 Diet & Lifestyle**
                    - Eat nutritious food
                    - Avoid processed foods
                    - Maintain good hygiene
                    """)

                # Download report
                st.markdown("---")
                st.markdown("### 📄 Download Report")
                
                report = f"""
=====================================
      AI DISEASE DIAGNOSIS REPORT
=====================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

-------------------------------------
      DIAGNOSIS RESULTS
-------------------------------------

Predicted Disease: {disease}
Confidence Score: {confidence:.2f}%
Risk Level: {'High' if confidence > 80 else 'Medium' if confidence > 60 else 'Low'}

-------------------------------------
      SYMPTOMS SELECTED
-------------------------------------
{', '.join(selected)}

-------------------------------------
      RECOMMENDATIONS
-------------------------------------
• Consult a qualified doctor
• Take prescribed medicines
• Get plenty of rest
• Stay hydrated
• Eat healthy food

=====================================
⚠️ This is an educational project.
   Always consult a healthcare professional.
=====================================
"""
                st.download_button(
                    label="📥 Download Diagnosis Report",
                    data=report,
                    file_name=f"diagnosis_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )

# ---------- ABOUT ----------
else:
    st.markdown("""
    ### ℹ️ About This System
    
    **AI-Based Disease Diagnosis and Recommendation System**
    
    Built with:
    - **Streamlit** – Web framework
    - **Scikit-learn** – Machine learning
    - **Random Forest** – Classification algorithm
    - **Python** – Programming language
    
    **Developer:** Thamizhmathi Sivakumar
    
    **College:** Arunai Engineering College
    
    **Department:** CSE (Third Year, 2026)
    
    ⚠️ **Disclaimer:** This is an educational project for preliminary diagnosis only.  
    **Always consult a healthcare professional for accurate medical advice.**
    """)

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("""
<div class="footer">
    ⚠️ <strong>Disclaimer:</strong> Educational purposes only. Not a substitute for professional medical advice.<br>
    © 2026 AI Disease Diagnosis System | Thamizhmathi Sivakumar
</div>
""", unsafe_allow_html=True)
