import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

## --- BACKEND LOGIC --- ##

def process_clinical_text(raw_text):
    """Uses NLTK to clean and extract key sentences."""
    if not raw_text: 
        return ""
    sentences = sent_tokenize(raw_text)
    # Filter for clinical keywords to reduce noise
    keywords = ['patient', 'history', 'diagnosis', 'pain', 'treatment', 'stable', 'denies']
    important = [s for s in sentences if any(k in s.lower() for k in keywords)]
    return " ".join(important[:5]) 

def analyze_vitals(df):
    """Identifies trends in vitals data."""
    summary = []
    if 'HeartRate' in df.columns:
        avg_hr = df['HeartRate'].mean()
        if avg_hr > 100: 
            summary.append(f"Tachycardic trend (Avg HR: {avg_hr:.1f})")
        elif avg_hr < 60: 
            summary.append(f"Bradycardic trend (Avg HR: {avg_hr:.1f})")
        else: 
            summary.append("Heart rate within normal limits.")
    return summary

## --- STREAMLIT UI --- ##

st.set_page_config(page_title="Multi-Modal Clinical Summarizer", layout="wide")
st.title("🏥 Clinical Multi-Modal Summarizer")
st.markdown("Combine **Notes + Labs + Vitals** into a single cohesive report.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Physician Notes")
    user_notes = st.text_area("Paste clinical notes here...", height=200, 
                             placeholder="Patient presents with acute chest pain...")

    st.subheader("2. Vitals Log (Time-Series)")
    # Sample data generator
    chart_data = pd.DataFrame(
        np.random.randint(60, 110, size=(10, 2)),
        columns=['HeartRate', 'SpO2']
    )
    st.line_chart(chart_data)

with col2:
    st.subheader("3. Lab Results")
    uploaded_file = st.file_uploader("Upload Lab CSV", type=["csv"])
    if uploaded_file:
        lab_df = pd.read_csv(uploaded_file)
        st.dataframe(lab_df)
    else:
        lab_df = pd.DataFrame({
            'Test': ['Hemoglobin', 'WBC', 'Creatinine'],
            'Result': [11.2, 12.5, 0.9],
            'Unit': ['g/dL', '10^3/uL', 'mg/dL'],
            'Status': ['Low', 'High', 'Normal']
        })
        st.table(lab_df)

st.divider() # This replaces the "---" that caused the error

if st.button("Generate Integrated Summary"):
    if not user_notes:
        st.warning("Please enter some clinical notes first.")
    else:
        with st.spinner("Synthesizing multi-modal data..."):
            # 1. Process Text with NLTK
            cleaned_text = process_clinical_text(user_notes)
            
            # 2. Process Vitals
            vital_trends = analyze_vitals(chart_data)
            
            # 3. Process Labs
            abnormal_labs = lab_df[lab_df['Status'] != 'Normal']['Test'].tolist()
            
            # FINAL DISPLAY
            st.success("### Final Clinical Summary")
            
            st.write(f"**Subjective/Objective (Text):** {cleaned_text}")
            
            st.write(f"**Lab Abnormalities:** {', '.join(abnormal_labs) if abnormal_labs else 'None'}")
            
            st.write("**Vital Trends:**")
            for trend in vital_trends:
                st.write(f"- {trend}")