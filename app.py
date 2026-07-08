import streamlit as st
import os
import sys
import traceback

st.set_page_config(page_title="Diagnostics", layout="wide")
st.title("🔍 File Check & Model Loader")

# List all files in current directory
st.subheader("📁 Files in current directory:")
files = os.listdir('.')
for f in files:
    st.write(f"- {f}")

# Now try to load the model with full error details
st.subheader("🔄 Attempting to load model...")

try:
    import joblib
    st.write("✅ joblib imported")
except Exception as e:
    st.error(f"❌ joblib import failed: {e}")
    st.stop()

# Function to load and show error
def safe_load(filename):
    try:
        obj = joblib.load(filename)
        st.success(f"✅ Loaded {filename}")
        return obj
    except Exception as e:
        st.error(f"❌ Failed to load {filename}")
        st.code(f"Exception: {e}\nType: {type(e)}\nTraceback:\n{traceback.format_exc()}", language="python")
        return None

model = safe_load('model.pkl')
scaler = safe_load('scaler.pkl')
encoder = safe_load('encoder.pkl')
features = safe_load('features.pkl')
diseases = safe_load('diseases.pkl')

if all([model, scaler, encoder, features, diseases]):
    st.success("🎉 All files loaded successfully! You can now deploy your full app.")
else:
    st.error("❌ Some files failed to load. Please check the errors above.")
