

import streamlit as st
import json
from my_pipeline import analyze_report   # import your pipeline function

st.set_page_config(page_title="Lab Report Analyzer", page_icon="🧪", layout="wide")

st.title("🧪 Lab Report Analyzer (LFT / RFT / Thyroid)")
st.write("Upload your **PDF or Image** lab test report. The app will detect report type, extract test values, predict risk, and give **Tamil suggestions**.")

# File uploader
uploaded_file = st.file_uploader("Upload Lab Report (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    # Save uploaded file temporarily
    with open("temp_report.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Run pipeline
    result = analyze_report("temp_report.pdf")

    # Display results
    st.subheader("📌 Report Summary")
    st.write(f"**Report Type:** {result['Report_Type']}")
    st.write(f"**Overall Diagnosis:** {result['Overall_Diagnosis']}")
    st.success(result["Tamil_Suggestion"])

    st.subheader("📊 Test Results")
    tests_df = []
    for t in result["Tests"]:
        tests_df.append(t)
    import pandas as pd
    st.dataframe(pd.DataFrame(tests_df))

    st.subheader("🔎 Debug Info")
    st.json(result, expanded=False)
