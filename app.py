import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(page_title="🏥 AI Disease Diagnosis", page_icon="🏥", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    encoder = joblib.load('models/encoder.pkl')
    features = joblib.load('models/features.pkl')
    diseases = joblib.load('models/diseases.pkl')
    return model, scaler, encoder, features, diseases

model, scaler, encoder, features, diseases = load_model()

with st.sidebar:
    st.markdown("### 🏥 Navigation")
    page = st.radio("", ["🏠 Home", "🩺 Disease Prediction", "📄 Project Report", "ℹ️ About"])
    st.markdown("---")
    st.markdown(f"**Diseases:** {len(diseases)}")
    st.markdown(f"**Symptoms:** {len(features)}")

if page == "🏠 Home":
    st.title("🏥 AI Disease Diagnosis System")
    st.markdown("""
    Welcome to the **AI-powered Disease Diagnosis System**.
    - Select your symptoms on the **Disease Prediction** page.
    - Get instant diagnosis with confidence.
    - View treatment recommendations.
    - Download your report.
    """)
    st.success("✅ Model loaded successfully. Ready for prediction!")

elif page == "🩺 Disease Prediction":
    st.title("🩺 Disease Prediction")
    st.write("Select all symptoms that apply:")

    selected_symptoms = []
    cols = st.columns(4)
    for i, sym in enumerate(features):
        col = cols[i % 4]
        # Display symptom name nicely (remove underscores)
        display_name = sym.replace('_', ' ').title()
        if col.checkbox(display_name, key=sym):
            selected_symptoms.append(sym)

    if st.button("🔍 Predict", type="primary", use_container_width=True):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            with st.spinner("Analyzing..."):
                # Build binary vector
                input_vec = np.zeros(len(features))
                for sym in selected_symptoms:
                    if sym in features:
                        input_vec[features.index(sym)] = 1
                input_scaled = scaler.transform([input_vec])

                pred_idx = model.predict(input_scaled)[0]
                disease = diseases[pred_idx]
                probs = model.predict_proba(input_scaled)[0]
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
                        st.error("🔴 High Risk - Consult a doctor immediately.")
                    elif confidence > 60:
                        st.warning("🟡 Medium Risk - Monitor symptoms.")
                    else:
                        st.success("🟢 Low Risk - Take rest and stay hydrated.")

                # Emergency detection – you can add more symptoms as needed
                emergency = ['chest_pain', 'shortness_breath', 'fainting', 'seizures']
                if any(s in emergency for s in selected_symptoms):
                    st.error("🚨 Emergency symptoms detected! Seek immediate medical attention.")

                st.subheader("💊 Recommendations")
                col1, col2, col3 = st.columns(3)
                col1.markdown("**Medications**\n- Consult a doctor\n- Take prescribed medicines")
                col2.markdown("**Home Care**\n- Rest\n- Hydrate\n- Monitor symptoms")
                col3.markdown("**Diet**\n- Balanced meals\n- Avoid processed food\n- Stay hygienic")

                report = f"""
=====================================
AI DIAGNOSIS REPORT
=====================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Disease: {disease}
Confidence: {confidence:.2f}%
Symptoms: {', '.join(selected_symptoms)}
=====================================
"""
                st.download_button("📄 Download Report", report, file_name=f"diagnosis_{datetime.now().strftime('%Y%m%d')}.txt", mime="text/plain")

elif page == "📄 Project Report":
    st.title("📄 Project Report")
    col1, col2, col3 = st.columns(3)
    col1.metric("Diseases", len(diseases))
    col2.metric("Symptoms", len(features))
    col3.metric("Accuracy", "~88% (example)")

    st.markdown("---")
    st.subheader("🎯 Problem Statement")
    st.markdown("""
    - Delayed diagnosis due to limited healthcare access.
    - Confusion among similar symptoms.
    - Need for instant preliminary guidance.
    """)
    st.subheader("💡 Solution")
    st.markdown("""
    - AI-powered real-time diagnosis using Random Forest.
    - Easy symptom selection and instant results.
    - Emergency detection and recommendations.
    """)
    st.subheader("🛠️ Technology Stack")
    st.table(pd.DataFrame({
        "Layer": ["Frontend", "Backend", "ML", "Data", "Deployment"],
        "Tech": ["Streamlit", "Python", "Scikit-learn", "Pandas/NumPy", "Render / Streamlit Cloud"]
    }))
    st.subheader("👨‍💻 Developer")
    st.markdown("""
    **Name:** Thamizhmathi Sivakumar  
    **College:** Arunai Engineering College  
    **Department:** CSE (Third Year, 2026)  
    **Email:** thamizhmathi477@gmail.com
    """)

else:
    st.title("ℹ️ About")
    st.markdown("""
    **AI-Based Disease Diagnosis and Recommendation System**  
    Built with Streamlit and Random Forest.  
    This is an educational project for preliminary diagnosis only.  
    **Always consult a healthcare professional.**
    """)
print("✅ app.py created.")
