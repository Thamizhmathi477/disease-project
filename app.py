import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import plotly.graph_objects as go
import plotly.express as px

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="🏥 MediAI - Disease Diagnosis System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- SESSION STATE INITIALIZATION ----------
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'history' not in st.session_state:
    st.session_state.history = []
if 'patient_name' not in st.session_state:
    st.session_state.patient_name = ""

# ---------- DARK MODE TOGGLE ----------
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
st.session_state.dark_mode = dark_mode

# ---------- CSS STYLING ----------
if dark_mode:
    st.markdown("""
    <style>
        /* Dark Mode */
        .stApp { background-color: #0a0e1a; }
        h1, h2, h3, h4, h5, h6, p, li, .stMarkdown, .stTextInput label, .stCheckbox label {
            color: #e0e0e0 !important;
        }
        .stTabs [data-baseweb="tab-list"] { background-color: #1a1a2e; border-radius: 10px; }
        .stTabs [data-baseweb="tab"] { background-color: #16213e; color: #e0e0e0 !important; border-color: #0f3460; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #1976D2; color: white !important; }
        .metric-card { background: #16213e !important; border-color: #0f3460 !important; }
        .metric-value { color: #4fc3f7 !important; }
        .metric-label { color: #a0a0a0 !important; }
        .diagnosis-box { background: linear-gradient(135deg, #0d1b2a, #1b2838) !important; border-left-color: #1976D2 !important; }
        .diagnosis-box h2 { color: #4fc3f7 !important; }
        .diagnosis-box p { color: #e0e0e0 !important; }
        .symptom-tag { background: #16213e !important; color: #4fc3f7 !important; border-color: #0f3460 !important; }
        .footer { border-top-color: #1a1a2e !important; color: #a0a0a0 !important; }
        .stTextInput input { background-color: #1a1a2e !important; color: #e0e0e0 !important; border-color: #1976D2 !important; }
        .stSelectbox select { background-color: #1a1a2e !important; color: #e0e0e0 !important; }
        .stDateInput input { background-color: #1a1a2e !important; color: #e0e0e0 !important; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        /* Light Mode */
        .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
        h1, h2, h3, h4, h5, h6, p, li, .stMarkdown { color: #1a1a2e !important; }
        .stTabs [data-baseweb="tab-list"] { background-color: #e8ecf0; border-radius: 10px; }
        .stTabs [data-baseweb="tab"] { background-color: white; color: #0D47A1 !important; border-color: #BBDEFB; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #1976D2; color: white !important; }
        .metric-card { background: white !important; border-color: #e8ecf0 !important; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
        .metric-value { color: #0D47A1 !important; }
        .metric-label { color: #6c757d !important; }
        .diagnosis-box { background: linear-gradient(135deg, #E3F2FD, #BBDEFB) !important; border-left-color: #1976D2 !important; }
        .diagnosis-box h2 { color: #0D47A1 !important; }
        .diagnosis-box p { color: #1a1a2e !important; }
        .symptom-tag { background: #E3F2FD !important; color: #0D47A1 !important; border-color: #BBDEFB !important; }
        .footer { border-top-color: #e8ecf0 !important; color: #6c757d !important; }
        .stTextInput input { border-color: #1976D2 !important; }
        .stSelectbox select { border-color: #1976D2 !important; }
        .stDateInput input { border-color: #1976D2 !important; }
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
    <div style="text-align:center; padding: 10px;">
        <h1 style="font-size:2.5rem; margin:0;">🏥</h1>
        <h2 style="margin:0; color:#1976D2;">MediAI</h2>
        <p style="margin:0; opacity:0.7;">AI Disease Diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Patient Information
    st.markdown("### 👤 Patient Details")
    st.session_state.patient_name = st.text_input("Name", placeholder="Enter patient name")
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])
    date = st.date_input("Date", datetime.now())

    st.markdown("---")

    # Model Info
    st.markdown("### 📊 Model Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🏥 Diseases", len(diseases))
    with col2:
        st.metric("🔬 Symptoms", len(features))
    st.metric("🎯 Accuracy", "88%")

    st.markdown("---")

    # Developer Info
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Thamizhmathi Sivakumar**")
    st.markdown("Arunai Engineering College")
    st.markdown("CSE · Third Year (2026)")

    st.markdown("---")

    # Quick Links
    st.markdown("### 🔗 Quick Links")
    st.markdown("[🌐 Live App](https://disease-project-r8ecaz25yrcwcr2dbkudnp.streamlit.app/)")
    st.markdown("[📂 GitHub](https://github.com/Thamizhmathi477/disease-project)")

# ---------- MAIN CONTENT ----------
st.title("🏥 MediAI – Disease Diagnosis System")
st.markdown("*AI-Powered Healthcare Assistant*")
st.markdown("---")

# ---------- METRICS ROW ----------
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(diseases)}</div><div class="metric-label">🏥 Diseases</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(features)}</div><div class="metric-label">🔬 Symptoms</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-value">88%</div><div class="metric-label">📊 Accuracy</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="metric-value">✅</div><div class="metric-label">🟢 Live</div></div>', unsafe_allow_html=True)
with col5:
    st.markdown('<div class="metric-card"><div class="metric-value">🌟</div><div class="metric-label">📈 Random Forest</div></div>', unsafe_allow_html=True)

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

    **MediAI** is an AI-powered disease diagnosis system that uses machine learning to predict diseases based on your symptoms.

    ### 🎯 How It Works
    <div style="display:flex; gap:20px; flex-wrap:wrap; margin:20px 0;">
        <div style="background:#E3F2FD; padding:15px; border-radius:10px; flex:1; min-width:150px; text-align:center;">
            <h2 style="margin:0;">1️⃣</h2>
            <p><strong>Search</strong><br>Find your symptoms</p>
        </div>
        <div style="background:#E3F2FD; padding:15px; border-radius:10px; flex:1; min-width:150px; text-align:center;">
            <h2 style="margin:0;">2️⃣</h2>
            <p><strong>Select</strong><br>Choose all that apply</p>
        </div>
        <div style="background:#E3F2FD; padding:15px; border-radius:10px; flex:1; min-width:150px; text-align:center;">
            <h2 style="margin:0;">3️⃣</h2>
            <p><strong>Predict</strong><br>Get instant diagnosis</p>
        </div>
        <div style="background:#E3F2FD; padding:15px; border-radius:10px; flex:1; min-width:150px; text-align:center;">
            <h2 style="margin:0;">4️⃣</h2>
            <p><strong>Report</strong><br>Download results</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Features grid
    st.markdown("### ✨ Key Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:white; padding:15px; border-radius:10px; border-left:4px solid #1976D2; margin:5px 0;">
            <strong>🏥 41 Diseases</strong><br>
            <span style="font-size:0.9rem; color:#6c757d;">Comprehensive coverage</span>
        </div>
        <div style="background:white; padding:15px; border-radius:10px; border-left:4px solid #1976D2; margin:5px 0;">
            <strong>🔬 131 Symptoms</strong><br>
            <span style="font-size:0.9rem; color:#6c757d;">Extensive symptom list</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:white; padding:15px; border-radius:10px; border-left:4px solid #43A047; margin:5px 0;">
            <strong>🎯 88% Accuracy</strong><br>
            <span style="font-size:0.9rem; color:#6c757d;">High prediction confidence</span>
        </div>
        <div style="background:white; padding:15px; border-radius:10px; border-left:4px solid #43A047; margin:5px 0;">
            <strong>⚡ Instant Results</strong><br>
            <span style="font-size:0.9rem; color:#6c757d;">Real-time diagnosis</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:white; padding:15px; border-radius:10px; border-left:4px solid #FB8C00; margin:5px 0;">
            <strong>📄 Reports</strong><br>
            <span style="font-size:0.9rem; color:#6c757d;">Downloadable diagnosis</span>
        </div>
        <div style="background:white; padding:15px; border-radius:10px; border-left:4px solid #FB8C00; margin:5px 0;">
            <strong>🌙 Dark Mode</strong><br>
            <span style="font-size:0.9rem; color:#6c757d;">Comfortable viewing</span>
        </div>
        """, unsafe_allow_html=True)

    # Disclaimer
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

    # Search bar
    search_term = st.text_input("🔍 Search Symptoms", placeholder="Type symptom name...")

    # Filter symptoms
    if search_term:
        filtered_features = [s for s in features if search_term.lower() in s.lower()]
    else:
        filtered_features = features

    st.caption(f"Showing {len(filtered_features)} of {len(features)} symptoms")

    # Symptom categories
    categories = {
        "🌡️ General": ["fever", "fatigue", "chills", "sweating", "weakness", "weight_loss"],
        "🫁 Respiratory": ["cough", "shortness_breath", "chest_pain", "sore_throat", "runny_nose"],
        "🤕 Pain": ["headache", "body_ache", "muscle_pain", "joint_pain", "back_pain"],
        "💊 Digestive": ["nausea", "vomiting", "diarrhea", "abdominal_pain", "loss_appetite"],
        "🧠 Neurological": ["dizziness", "confusion", "numbness", "tingling", "tremors"],
        "🩸 Skin": ["rash", "itching", "hives", "dry_skin", "jaundice"]
    }

    selected = []

    # Show symptoms
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

    # Show selected tags
    if selected:
        st.markdown("#### 📋 Selected Symptoms:")
        tags = "".join([f'<span class="symptom-tag">✅ {s.replace("_", " ").title()}</span>' for s in selected])
        st.markdown(tags, unsafe_allow_html=True)

    # Predict button
    if st.button("🔍 Predict Disease", type="primary"):
        if not selected:
            st.warning("⚠️ Please select at least one symptom.")
        else:
            with st.spinner("🧠 Analyzing symptoms..."):
                # Progress bar
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

                # ---------- RESULTS ----------
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

                # Confidence Gauge
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

                # Emergency detection
                emergency = ['chest_pain', 'shortness_breath', 'fainting', 'seizures']
                if any(s in emergency for s in selected):
                    st.error("🚨 **Emergency symptoms detected!** Seek immediate medical attention.")

                # Doctor recommendation
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
                    'Hypertension': 'Cardiologist',
                    'Malaria': 'Infectious Disease Specialist',
                    'Typhoid': 'Infectious Disease Specialist',
                    'Tuberculosis': 'Pulmonologist',
                    'Pneumonia': 'Pulmonologist',
                    'Bronchitis': 'Pulmonologist',
                    'Common Cold': 'General Physician',
                    'Influenza': 'General Physician'
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
                    - Follow dosage instructions
                    """)
                with rec_col2:
                    st.markdown("""
                    **🏠 Home Care**
                    - Get plenty of rest
                    - Stay hydrated
                    - Monitor symptoms
                    - Maintain hygiene
                    """)
                with rec_col3:
                    st.markdown("""
                    **🥗 Diet & Lifestyle**
                    - Eat nutritious food
                    - Avoid processed foods
                    - Exercise regularly
                    - Sleep 7-8 hours
                    """)

                st.markdown("---")
                st.markdown("### 📄 Download Report")

                report = f"""
=====================================
         MEDIAI DIAGNOSIS REPORT
=====================================

Patient Name: {st.session_state.patient_name if st.session_state.patient_name else 'Not Provided'}
Age: {age}
Gender: {gender}
Date: {date.strftime('%Y-%m-%d')}

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

                # Save to history
                st.session_state.history.append({
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'patient': st.session_state.patient_name if st.session_state.patient_name else 'Unknown',
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
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
            'Score': ['88%', '89%', '87%', '88%', '0.91']
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
        | n_jobs | -1 |

        **Why Random Forest?**
        - ✅ Ensemble method
        - ✅ Reduces overfitting
        - ✅ Handles high-dimensional data
        - ✅ Feature importance available
        - ✅ Fast prediction
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

    # Confusion Matrix
    st.markdown("### 📊 Confusion Matrix")
    st.markdown("""
    *The confusion matrix shows how well the model performs across all diseases.*
    *Green = correct predictions, Red = incorrect predictions.*
    """)

    # Feature Distribution
    # Feature Distribution (using matplotlib)
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

        # Download history
        if st.button("📥 Download History as CSV"):
            csv = history_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                data=csv,
                file_name="diagnosis_history.csv",
                mime="text/csv"
            )

    st.markdown("---")
    st.markdown("### 📊 Summary Statistics")
    if st.session_state.history:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Diagnoses", len(st.session_state.history))
        with col2:
            diseases_list = [h['disease'] for h in st.session_state.history]
            most_common = max(set(diseases_list), key=diseases_list.count) if diseases_list else "N/A"
            st.metric("Most Common Disease", most_common)
        with col3:
            avg_confidence = np.mean([float(h['confidence'].replace('%', '')) for h in st.session_state.history])
            st.metric("Average Confidence", f"{avg_confidence:.1f}%")

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
    | **Version Control** | GitHub |

    ### 👨‍💻 Developer

    **Thamizhmathi Sivakumar**
    - 3rd Year CSE Student
    - Arunai Engineering College
    - 📧 thamizhmathi477@gmail.com
    - 📍 Thiruvannamalai

    ### 🔗 Links
    - 🌐 [Live App](https://disease-project-r8ecaz25yrcwcr2dbkudnp.streamlit.app/)
    - 📂 [GitHub Repository](https://github.com/Thamizhmathi477/disease-project)
    - 🔗 [LinkedIn](https://linkedin.com/in/Thamizhmathi Sivakumar)

    ### 📚 References
    1. Dahiwade, D., et al. (2019). "Designing Disease Prediction Model Using Machine Learning Approach." *ICCMC*.
    2. Grampurohit, S., et al. (2020). "Disease Prediction using Machine Learning Algorithms." *INCET*.
    3. Shetty, S.V., et al. (2019). "Symptom Based Health Prediction using Data Mining." *ICCES*.
    4. Chen, M., et al. (2017). "Disease Prediction by Machine Learning Over Big Data." *IEEE Access*.
    """)

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(f"""
<div class="footer">
    ⚠️ <strong>Disclaimer:</strong> Educational purposes only. Not a substitute for professional medical advice.<br>
    © 2026 MediAI | Developed by Thamizhmathi Sivakumar | Arunai Engineering College
</div>
""", unsafe_allow_html=True)
