import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import plotly.graph_objects as go

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="🏥 MediAI - Disease Diagnosis System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- SCROLL FIX – CSS ----------
st.markdown("""
<style>
    /* Enable scrolling for the entire app */
    .main {
        overflow-y: auto !important;
        height: 100vh !important;
    }
    
    /* Ensure content container can scroll */
    .block-container {
        overflow-y: auto !important;
        max-height: 100vh !important;
        padding-bottom: 100px !important;
    }
    
    /* Sidebar scrolling */
    section[data-testid="stSidebar"] {
        overflow-y: auto !important;
        max-height: 100vh !important;
    }
    
    /* Custom scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f0f0f0;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #1976D2;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #0D47A1;
    }
    
    /* Dark mode scrollbar */
    @media (prefers-color-scheme: dark) {
        ::-webkit-scrollbar-track {
            background: #1a1a2e;
        }
        ::-webkit-scrollbar-thumb {
            background: #1976D2;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'history' not in st.session_state:
    st.session_state.history = []

# ---------- DARK MODE TOGGLE ----------
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
st.session_state.dark_mode = dark_mode

# ---------- CSS STYLING - PREMIUM HOSPITAL THEME ----------
if dark_mode:
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #0a0e1a, #1a1a2e, #0a1628); }
        h1, h2, h3, h4, h5, h6, p, li, .stMarkdown, .stTextInput label, .stCheckbox label {
            color: #e0e0e0 !important;
        }
        .stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.05); border-radius: 12px; backdrop-filter: blur(10px); }
        .stTabs [data-baseweb="tab"] { background: transparent; color: #a0a0a0 !important; border: none; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background: #1976D2; color: white !important; border-radius: 8px; }
        .metric-card { background: rgba(255,255,255,0.05) !important; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 16px !important; }
        .metric-value { color: #4fc3f7 !important; }
        .metric-label { color: #a0a0a0 !important; }
        .diagnosis-box { background: linear-gradient(135deg, rgba(13,71,161,0.2), rgba(25,118,210,0.1)) !important; border: 1px solid rgba(25,118,210,0.2) !important; }
        .diagnosis-box h2 { color: #4fc3f7 !important; }
        .diagnosis-box p { color: #e0e0e0 !important; }
        .symptom-tag { background: rgba(25,118,210,0.2) !important; color: #4fc3f7 !important; border: 1px solid rgba(25,118,210,0.2) !important; }
        .footer { border-top-color: rgba(255,255,255,0.05) !important; color: #a0a0a0 !important; }
        .stTextInput input { background: rgba(255,255,255,0.05) !important; color: #e0e0e0 !important; border: 1px solid rgba(25,118,210,0.3) !important; }
        .stButton button { background: linear-gradient(135deg, #1976D2, #0D47A1) !important; color: white !important; border: none !important; }
        .hospital-header { background: linear-gradient(135deg, rgba(13,71,161,0.3), rgba(25,118,210,0.1)) !important; backdrop-filter: blur(10px); border: 1px solid rgba(25,118,210,0.1) !important; }
        .risk-high { background: rgba(211,47,47,0.15) !important; border-left: 4px solid #D32F2F !important; color: #FF6B6B !important; }
        .risk-medium { background: rgba(245,124,0,0.15) !important; border-left: 4px solid #F57C00 !important; color: #FFB74D !important; }
        .risk-low { background: rgba(56,142,60,0.15) !important; border-left: 4px solid #388E3C !important; color: #81C784 !important; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #e8f0fe 0%, #f5f7fa 50%, #e3ecf5 100%);
        }
        h1, h2, h3, h4, h5, h6 { 
            color: #0D47A1 !important; 
            font-weight: 700 !important;
        }
        p, li, .stMarkdown { color: #1a1a2e !important; }
        .stTabs [data-baseweb="tab-list"] { 
            background: rgba(255,255,255,0.7);
            backdrop-filter: blur(20px);
            border-radius: 14px; 
            padding: 6px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            border: 1px solid rgba(255,255,255,0.5);
        }
        .stTabs [data-baseweb="tab"] { 
            background: transparent; 
            color: #0D47A1 !important; 
            border-radius: 10px;
            font-weight: 500;
            padding: 8px 20px;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { 
            background: #1976D2; 
            color: white !important; 
            box-shadow: 0 4px 12px rgba(25,118,210,0.3);
        }
        .metric-card { 
            background: rgba(255,255,255,0.8) !important;
            backdrop-filter: blur(20px);
            border-radius: 16px !important;
            padding: 22px !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06) !important;
            border: 1px solid rgba(255,255,255,0.6) !important;
            transition: all 0.3s ease;
        }
        .metric-card:hover { 
            transform: translateY(-6px);
            box-shadow: 0 12px 40px rgba(13,71,161,0.12) !important;
        }
        .metric-value { 
            color: #0D47A1 !important; 
            font-size: 2.2rem !important;
            font-weight: 700 !important;
        }
        .metric-label { 
            color: #6c757d !important; 
            font-size: 0.9rem !important;
            font-weight: 500 !important;
        }
        .diagnosis-box { 
            background: linear-gradient(135deg, rgba(227, 242, 253, 0.9), rgba(187, 222, 251, 0.7)) !important;
            backdrop-filter: blur(10px);
            border-radius: 16px !important;
            padding: 25px !important;
            border-left: 6px solid #1976D2 !important;
            box-shadow: 0 8px 32px rgba(25,118,210,0.15) !important;
            border: 1px solid rgba(255,255,255,0.5);
        }
        .diagnosis-box h2 { color: #0D47A1 !important; font-size: 32px !important; }
        .diagnosis-box p { color: #1a1a2e !important; font-size: 18px !important; }
        .risk-high { 
            background: rgba(255,235,238,0.9) !important;
            backdrop-filter: blur(10px);
            color: #D32F2F !important; 
            padding: 14px 20px; 
            border-radius: 12px; 
            font-weight: 700; 
            font-size: 18px; 
            border-left: 5px solid #D32F2F;
            box-shadow: 0 2px 12px rgba(211,47,47,0.12);
            border: 1px solid rgba(211,47,47,0.1);
        }
        .risk-medium { 
            background: rgba(255,243,224,0.9) !important;
            backdrop-filter: blur(10px);
            color: #F57C00 !important; 
            padding: 14px 20px; 
            border-radius: 12px; 
            font-weight: 700; 
            font-size: 18px; 
            border-left: 5px solid #F57C00;
            box-shadow: 0 2px 12px rgba(245,124,0,0.12);
            border: 1px solid rgba(245,124,0,0.1);
        }
        .risk-low { 
            background: rgba(232,245,233,0.9) !important;
            backdrop-filter: blur(10px);
            color: #388E3C !important; 
            padding: 14px 20px; 
            border-radius: 12px; 
            font-weight: 700; 
            font-size: 18px; 
            border-left: 5px solid #388E3C;
            box-shadow: 0 2px 12px rgba(56,142,60,0.12);
            border: 1px solid rgba(56,142,60,0.1);
        }
        .symptom-tag { 
            background: rgba(227, 242, 253, 0.8) !important; 
            color: #0D47A1 !important; 
            padding: 6px 16px; 
            border-radius: 20px; 
            font-size: 14px; 
            margin: 4px; 
            display: inline-block; 
            border: 1px solid rgba(25,118,210,0.15) !important;
            font-weight: 500;
        }
        .stButton button { 
            background: linear-gradient(135deg, #1976D2, #0D47A1) !important; 
            color: white !important; 
            font-size: 18px !important; 
            font-weight: 600 !important; 
            border-radius: 12px !important; 
            padding: 14px 30px !important; 
            border: none !important; 
            width: 100% !important;
            box-shadow: 0 4px 16px rgba(25,118,210,0.3) !important;
            transition: all 0.3s ease !important;
        }
        .stButton button:hover { 
            transform: translateY(-2px) scale(1.01);
            box-shadow: 0 8px 32px rgba(25,118,210,0.4) !important;
        }
        .stTextInput input { 
            border: 2px solid rgba(25,118,210,0.2) !important; 
            border-radius: 12px !important;
            padding: 14px !important;
            font-size: 16px !important;
            background: rgba(255,255,255,0.7) !important;
            backdrop-filter: blur(10px);
        }
        .stTextInput input:focus { 
            border-color: #1976D2 !important; 
            box-shadow: 0 0 0 4px rgba(25,118,210,0.12) !important;
        }
        .footer { 
            text-align: center; 
            color: #6c757d; 
            padding: 20px; 
            margin-top: 30px; 
            font-size: 14px;
            background: rgba(255,255,255,0.7);
            backdrop-filter: blur(20px);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.5);
        }
        section[data-testid="stSidebar"] {
            background: rgba(255,255,255,0.85) !important;
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255,255,255,0.5);
        }
        .hospital-header {
            background: linear-gradient(135deg, #0D47A1, #1565C0, #1976D2);
            padding: 25px 35px;
            border-radius: 18px;
            color: white;
            margin-bottom: 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 8px 32px rgba(13,71,161,0.3);
            border: 1px solid rgba(255,255,255,0.1);
        }
        .hospital-header h1 { 
            color: white !important; 
            margin: 0; 
            font-size: 2rem;
            font-weight: 700;
        }
        .hospital-header p { 
            color: rgba(255,255,255,0.9) !important; 
            margin: 0;
        }
        .hospital-badge {
            background: rgba(255,255,255,0.2);
            padding: 6px 18px;
            border-radius: 20px;
            color: white !important;
            font-size: 0.8rem;
            font-weight: 500;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        .hospital-footer {
            background: rgba(255,255,255,0.7);
            backdrop-filter: blur(20px);
            padding: 18px;
            border-radius: 12px;
            text-align: center;
            color: #6c757d;
            margin-top: 30px;
            border: 1px solid rgba(255,255,255,0.5);
        }
        .stCheckbox label { font-size: 15px !important; color: #1a1a2e !important; font-weight: 500 !important; }
        .stProgress > div > div {
            background: linear-gradient(90deg, #1976D2, #0D47A1) !important;
            border-radius: 10px !important;
        }
        .feature-card {
            background: rgba(255,255,255,0.7);
            backdrop-filter: blur(10px);
            padding: 15px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.5);
            margin: 8px 0;
            transition: all 0.3s;
        }
        .feature-card:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        }
        .step-card {
            background: rgba(255,255,255,0.7);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 2px 12px rgba(0,0,0,0.04);
            border: 1px solid rgba(255,255,255,0.5);
            transition: all 0.3s;
        }
        .step-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        }
        .stTable {
            background: rgba(255,255,255,0.7) !important;
            backdrop-filter: blur(10px);
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,0.5) !important;
        }
    </style>
    """, unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    features = joblib.load('features.pkl')
    diseases = joblib.load('diseases.pkl')
    return model, scaler, encoder, features, diseases

try:
    model, scaler, encoder, features, diseases = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <div style="font-size:3.5rem;">🏥</div>
        <h2 style="color:#0D47A1; margin:0; font-weight:700;">MediAI</h2>
        <p style="color:#6c757d; margin:0; font-size:0.9rem;">AI Disease Diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📊 Model Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🏥 Diseases", len(diseases))
    with col2:
        st.metric("🔬 Symptoms", len(features))
    st.metric("🎯 Accuracy", "88%")

    st.markdown("---")

    st.markdown("### 🚀 Quick Links")
    st.markdown("[🌐 Live App](https://disease-project-r8ecaz25yrcwcr2dbkudnp.streamlit.app/)")
    st.markdown("[📂 GitHub](https://github.com/Thamizhmathi477/disease-project)")

    st.markdown("---")

    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Thamizhmathi Sivakumar**")
    st.markdown("Arunai Engineering College")
    st.markdown("CSE · Third Year (2026)")
    st.markdown("📍 Thiruvannamalai")

# ---------- HOSPITAL HEADER ----------
st.markdown("""
<div class="hospital-header">
    <div>
        <h1>🏥 MediAI</h1>
        <p>AI-Powered Disease Diagnosis System</p>
    </div>
    <div>
        <span class="hospital-badge">🟢 Live</span>
        <span class="hospital-badge" style="margin-left:8px;">v2.0</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- METRICS ROW ----------
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(diseases)}</div><div class="metric-label">🏥 Diseases</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(features)}</div><div class="metric-label">🔬 Symptoms</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-value">88%</div><div class="metric-label">📊 Accuracy</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="metric-value">⚡</div><div class="metric-label">Real-Time</div></div>', unsafe_allow_html=True)
with col5:
    st.markdown('<div class="metric-card"><div class="metric-value">🛡️</div><div class="metric-label">AI Powered</div></div>', unsafe_allow_html=True)

# ---------- TABS ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home",
    "🩺 Disease Prediction",
    "📊 Model Insights",
    "📋 History",
    "ℹ️ About"
])

# ============================================
# TAB 1: HOME
# ============================================
with tab1:
    st.markdown("""
    ## 👋 Welcome to MediAI

    **MediAI** is an intelligent healthcare assistant that uses machine learning to predict diseases from symptoms.

    ### 🎯 How It Works
    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(180px,1fr)); gap:15px; margin:20px 0;">
        <div class="step-card">
            <div style="font-size:2.5rem;">1️⃣</div>
            <strong>Search</strong>
            <p style="font-size:0.9rem; color:#6c757d;">Find your symptoms</p>
        </div>
        <div class="step-card">
            <div style="font-size:2.5rem;">2️⃣</div>
            <strong>Select</strong>
            <p style="font-size:0.9rem; color:#6c757d;">Choose all that apply</p>
        </div>
        <div class="step-card">
            <div style="font-size:2.5rem;">3️⃣</div>
            <strong>Predict</strong>
            <p style="font-size:0.9rem; color:#6c757d;">Get instant diagnosis</p>
        </div>
        <div class="step-card">
            <div style="font-size:2.5rem;">4️⃣</div>
            <strong>Report</strong>
            <p style="font-size:0.9rem; color:#6c757d;">Download results</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ✨ Key Features")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <strong>🏥 41 Diseases</strong><br>
            <span style="font-size:0.9rem; color:#6c757d;">Comprehensive disease coverage</span>
        </div>
        <div class="feature-card">
            <strong>🔬 131 Symptoms</strong><br>
            <span style="font-size:0.9rem; color:#6c757d;">Extensive symptom library</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <strong>🎯 88% Accuracy</strong><br>
            <span style="font-size:0.9rem; color:#6c757d;">High prediction confidence</span>
        </div>
        <div class="feature-card">
            <strong>⚡ Real-Time Results</strong><br>
            <span style="font-size:0.9rem; color:#6c757d;">Instant diagnosis in seconds</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <strong>📄 Download Reports</strong><br>
            <span style="font-size:0.9rem; color:#6c757d;">Save diagnosis results</span>
        </div>
        <div class="feature-card">
            <strong>🌙 Dark Mode</strong><br>
            <span style="font-size:0.9rem; color:#6c757d;">Comfortable viewing</span>
        </div>
        """, unsafe_allow_html=True)

    st.warning("""
    ⚠️ **Disclaimer:** This is an educational project for preliminary diagnosis only.  
    Always consult a healthcare professional for accurate medical advice.
    """)

# ============================================
# TAB 2: DISEASE PREDICTION
# ============================================
with tab2:
    st.markdown("## 🩺 Disease Prediction")
    st.markdown("### Select all symptoms that apply:")

    search_term = st.text_input("🔍 Search Symptoms", placeholder="Type symptom name...")

    if search_term:
        filtered_features = [s for s in features if search_term.lower() in s.lower()]
    else:
        filtered_features = features

    st.caption(f"Showing {len(filtered_features)} of {len(features)} symptoms")

    categories = {
        "🌡️ General": ["fever", "fatigue", "chills", "sweating", "weakness", "weight_loss"],
        "🫁 Respiratory": ["cough", "shortness_breath", "chest_pain", "sore_throat", "runny_nose"],
        "🤕 Pain": ["headache", "body_ache", "muscle_pain", "joint_pain", "back_pain"],
        "💊 Digestive": ["nausea", "vomiting", "diarrhea", "abdominal_pain", "loss_appetite"],
        "🧠 Neurological": ["dizziness", "confusion", "numbness", "tingling", "tremors"],
        "🩸 Skin": ["rash", "itching", "hives", "dry_skin", "jaundice"]
    }

    selected = []

    if search_term:
        cols = st.columns(4)
        for i, sym in enumerate(filtered_features):
            col = cols[i % 4]
            if col.checkbox(sym.replace('_', ' ').title(), key=f"search_{sym}"):
                selected.append(sym)
    else:
        for category, sym_list in categories.items():
            category_symptoms = [s for s in sym_list if s in features]
            if category_symptoms:
                st.markdown(f"#### {category}")
                cols = st.columns(4)
                for i, sym in enumerate(category_symptoms):
                    col = cols[i % 4]
                    if col.checkbox(sym.replace('_', ' ').title(), key=f"cat_{sym}"):
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
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.005)
                    progress_bar.progress(i + 1)

                vec = np.zeros(len(features))
                for sym in selected:
                    if sym in features:
                        vec[features.index(sym)] = 1
                scaled = scaler.transform([vec])
                pred = model.predict(scaled)[0]
                disease = diseases[pred]
                probs = model.predict_proba(scaled)[0]
                confidence = max(probs) * 100

                progress_bar.empty()

                st.markdown("---")
                st.markdown("## 📊 Diagnosis Result")

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

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    title={'text': "Confidence Score"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1},
                        'bar': {'color': "#1976D2"},
                        'steps': [
                            {'range': [0, 40], 'color': "#FFEBEE"},
                            {'range': [40, 70], 'color': "#FFF3E0"},
                            {'range': [70, 100], 'color': "#E8F5E9"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)

                emergency = ['chest_pain', 'shortness_breath', 'fainting', 'seizures']
                if any(s in emergency for s in selected):
                    st.error("🚨 **Emergency symptoms detected!** Seek immediate medical attention.")

                doctor_mapping = {
                    'Heart Attack': 'Cardiologist',
                    'Dengue': 'Infectious Disease Specialist',
                    'Diabetes': 'Endocrinologist',
                    'Arthritis': 'Rheumatologist',
                    'Migraine': 'Neurologist',
                    'Asthma': 'Pulmonologist',
                    'Fungal Infection': 'Dermatologist',
                    'Acne': 'Dermatologist',
                    'Gastroenteritis': 'Gastroenterologist',
                    'Peptic Ulcer Disease': 'Gastroenterologist',
                    'Hypothyroidism': 'Endocrinologist',
                    'Hyperthyroidism': 'Endocrinologist',
                    'Hypertension': 'Cardiologist'
                }

                if disease in doctor_mapping:
                    st.info(f"👨‍⚕️ **Recommended Specialist:** {doctor_mapping[disease]}")

                st.markdown("---")
                st.markdown("### 💊 Recommendations")

                rec_col1, rec_col2, rec_col3 = st.columns(3)
                with rec_col1:
                    st.markdown("""
                    **💊 Medications**
                    - Consult a doctor
                    - Take prescribed medicines
                    - Do not self-medicate
                    """)
                with rec_col2:
                    st.markdown("""
                    **🏠 Home Care**
                    - Get plenty of rest
                    - Stay hydrated
                    - Monitor symptoms
                    """)
                with rec_col3:
                    st.markdown("""
                    **🥗 Diet & Lifestyle**
                    - Eat nutritious food
                    - Avoid processed foods
                    - Maintain good hygiene
                    """)

                st.markdown("---")
                st.markdown("### 📄 Download Report")

                report = f"""
=====================================
         MEDIAI DIAGNOSIS REPORT
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

{f'• Recommended Specialist: {doctor_mapping[disease]}' if disease in doctor_mapping else ''}

=====================================
⚠️ This is an educational project.
   Always consult a healthcare professional.
=====================================
"""
                st.download_button(
                    "📥 Download Diagnosis Report",
                    data=report,
                    file_name=f"diagnosis_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )

                st.session_state.history.append({
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'symptoms': ', '.join(selected),
                    'disease': disease,
                    'confidence': f"{confidence:.2f}%"
                })

# ============================================
# TAB 3: MODEL INSIGHTS
# ============================================
with tab3:
    st.markdown("## 📊 Model Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Model Performance")
        performance_data = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Score': ['88%', '89%', '87%', '88%']
        })
        st.table(performance_data)

    with col2:
        st.markdown("### 📈 Algorithm Details")
        st.markdown("""
        **Random Forest Classifier**

        | Parameter | Value |
        |-----------|-------|
        | n_estimators | 50 |
        | max_depth | 15 |
        | random_state | 42 |

        **Why Random Forest?**
        - ✅ Ensemble method
        - ✅ Reduces overfitting
        - ✅ Handles high-dimensional data
        - ✅ Feature importance available
        """)

    st.markdown("---")
    st.markdown("### 🔬 Top 15 Most Important Symptoms")

    importances = model.feature_importances_
    feature_names = features

    importance_df = pd.DataFrame({
        'Symptom': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importance_df)))
    ax.barh(importance_df['Symptom'], importance_df['Importance'], color=colors)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Top 15 Most Important Symptoms', fontsize=14)
    ax.tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### 📋 Symptom Distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    top_10 = importance_df.head(10)
    colors2 = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_10)))
    ax2.bar(top_10['Symptom'], top_10['Importance'], color=colors2)
    ax2.set_xlabel('Symptom', fontsize=12)
    ax2.set_ylabel('Importance', fontsize=12)
    ax2.set_title('Top 10 Symptoms by Importance', fontsize=14)
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    plt.tight_layout()
    st.pyplot(fig2)

# ============================================
# TAB 4: HISTORY
# ============================================
with tab4:
    st.markdown("## 📋 Diagnosis History")

    if not st.session_state.history:
        st.info("No diagnosis history yet. Go to the **Disease Prediction** tab to make your first prediction.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        if st.button("📥 Download History as CSV"):
            csv = history_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                data=csv,
                file_name="diagnosis_history.csv",
                mime="text/csv"
            )

# ============================================
# TAB 5: ABOUT
# ============================================
with tab5:
    st.markdown("## ℹ️ About MediAI")

    st.markdown("""
    ### 📌 Project Overview

    **MediAI** is an AI-powered disease diagnosis system developed as a third-year CSE project. It uses machine learning to predict diseases from patient symptoms.

    ### 🛠️ Technology Stack

    | Layer | Technology |
    |-------|------------|
    | **Frontend** | Streamlit |
    | **Backend** | Python 3.11 |
    | **Machine Learning** | Scikit-learn (Random Forest) |
    | **Data Processing** | Pandas, NumPy |
    | **Model Serialization** | Joblib |
    | **Deployment** | Streamlit Cloud |

    ### 👨‍💻 Developer

    **Thamizhmathi Sivakumar**
    - 3rd Year CSE Student
    - Arunai Engineering College
    - 📧 thamizhmathi477@gmail.com
    - 📍 Thiruvannamalai

    ### 🔗 Links
    - 🌐 [Live App](https://disease-project-r8ecaz25yrcwcr2dbkudnp.streamlit.app/)
    - 📂 [GitHub Repository](https://github.com/Thamizhmathi477/disease-project)

    ### 📚 References
    1. Dahiwade, D., et al. (2019). "Designing Disease Prediction Model Using Machine Learning Approach." *ICCMC*.
    2. Grampurohit, S., et al. (2020). "Disease Prediction using Machine Learning Algorithms." *INCET*.
    3. Shetty, S.V., et al. (2019). "Symptom Based Health Prediction using Data Mining." *ICCES*.
    4. Chen, M., et al. (2017). "Disease Prediction by Machine Learning Over Big Data." *IEEE Access*.
    """)

# ---------- HOSPITAL FOOTER ----------
st.markdown("""
<div class="hospital-footer">
    ⚠️ <strong>Disclaimer:</strong> Educational purposes only. Not a substitute for professional medical advice.<br>
    © 2026 MediAI | Developed by Thamizhmathi Sivakumar | Arunai Engineering College
</div>
""", unsafe_allow_html=True)
