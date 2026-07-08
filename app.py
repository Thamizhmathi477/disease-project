import streamlit as st
import os
import pickle
import joblib
import traceback

st.set_page_config(page_title="Diagnostics", layout="wide")
st.title("🔍 File Check & Model Loader")

# List all files
st.subheader("📁 Files in current directory:")
files = os.listdir('.')
for f in files:
    st.write(f"- {f}")

st.subheader("🔄 Attempting to load model...")

# Load model.pkl with pickle (not joblib)
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    st.success("✅ model.pkl loaded with pickle")
except Exception as e:
    st.error(f"❌ Failed to load model.pkl with pickle")
    st.code(f"Exception: {e}\nTraceback:\n{traceback.format_exc()}", language="python")

# Load other files with joblib
def safe_load_joblib(filename):
    try:
        obj = joblib.load(filename)
        st.success(f"✅ Loaded {filename}")
        return obj
    except Exception as e:
        st.error(f"❌ Failed to load {filename}")
        st.code(f"Exception: {e}", language="python")
        return None

scaler = safe_load_joblib('scaler.pkl')
encoder = safe_load_joblib('encoder.pkl')
features = safe_load_joblib('features.pkl')
diseases = safe_load_joblib('diseases.pkl')

if 'model' in locals() and all([scaler, encoder, features, diseases]):
    st.success("🎉 All files loaded successfully! You can now deploy your full app.")
else:
    st.error("❌ Some files failed to load. Please check the errors above.")
