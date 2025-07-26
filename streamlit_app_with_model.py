
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("cleaned_prostate_treatment.csv")

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🔁 Recurrence Predictor"])

# ========== TAB 1: DASHBOARD ==========
with tab1:
    st.title("Dashboard")
    st.write("A visual summary of prostate cancer treatment outcomes")

    if st.checkbox("Show raw data"):
        st.dataframe(df)

    # Gleason Score Distribution
    st.subheader("Distribution of Gleason Scores")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='Gleason_Score', order=sorted(df['Gleason_Score'].unique()), ax=ax1)
    ax1.set_ylabel("Patient Count")
    st.pyplot(fig1)

    # Side Effect Severity by Treatment
    st.subheader("Average Side Effect Severity by Treatment")
    severity_df = df.groupby('Treatment_Type')[['Urinary_Score', 'ED_Score']].mean().reset_index()
    st.dataframe(severity_df)

    fig2, ax2 = plt.subplots()
    severity_df.plot(kind='bar', x='Treatment_Type', ax=ax2)
    ax2.set_ylabel("Severity (0=None, 2=Severe)")
    st.pyplot(fig2)

# ========== TAB 2: RECURRENCE PREDICTOR ==========
with tab2:
    st.title("🔁 Recurrence Risk Predictor")
    st.write("Input patient features to estimate recurrence probability")

    # Prepare data
    df_model = df.dropna(subset=['Gleason_Numeric', 'PSA_Baseline', 'PSA_6_Months',
                                 'Urinary_Score', 'ED_Score', 'Recurrence_Flag_Binary'])

    X = df_model[['Gleason_Numeric', 'PSA_Baseline', 'PSA_6_Months', 'Urinary_Score', 'ED_Score']]
    y = df_model['Recurrence_Flag_Binary']
    model = LogisticRegression()
    model.fit(X, y)

    # User input
    st.subheader("Enter Patient Details")
    gleason = st.slider("Gleason Score (6–10)", 6, 10, 7)
    psa_baseline = st.number_input("PSA Baseline", min_value=0.0, value=10.0, step=0.1)
    psa_6m = st.number_input("PSA at 6 Months", min_value=0.0, value=1.0, step=0.1)
    urinary = st.selectbox("Urinary Incontinence Severity", options=[0, 1, 2], format_func=lambda x: ['None', 'Mild', 'Severe'][x])
    ed = st.selectbox("Erectile Dysfunction Severity", options=[0, 1, 2], format_func=lambda x: ['None', 'Mild', 'Severe'][x])

    if st.button("Predict Recurrence Probability"):
        input_df = pd.DataFrame([[gleason, psa_baseline, psa_6m, urinary, ed]],
                                columns=['Gleason_Numeric', 'PSA_Baseline', 'PSA_6_Months', 'Urinary_Score', 'ED_Score'])
        prob = model.predict_proba(input_df)[0][1]
        st.success(f"Estimated probability of recurrence: {prob:.2%}")
        
        from fpdf import FPDF
import io

# --- Create summary DataFrame ---
summary_df = pd.DataFrame([{
    "Gleason Score": gleason_score,
    "PSA Baseline": psa_baseline,
    "PSA at 6 Months": psa_6mo,
    "Urinary Severity": urinary_severity,
    "ED Severity": ed_severity,
    "Recurrence Probability": f"{recurrence_prob:.2%}"
}])

st.subheader("Summary of Prediction")
st.dataframe(summary_df)

# --- CSV Download ---
csv = summary_df.to_csv(index=False).encode("utf-8")
st.download_button("📄 Download Summary as CSV", data=csv, file_name="recurrence_summary.csv", mime="text/csv")

# --- PDF Download ---
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Prostate Cancer Recurrence Prediction Summary", ln=True, align='C')
pdf.ln(10)
pdf.cell(200, 10, txt=f"Gleason Score: {gleason_score}", ln=True)
pdf.cell(200, 10, txt=f"PSA Baseline: {psa_baseline}", ln=True)
pdf.cell(200, 10, txt=f"PSA at 6 Months: {psa_6mo}", ln=True)
pdf.cell(200, 10, txt=f"Urinary Incontinence Severity: {urinary_severity}", ln=True)
pdf.cell(200, 10, txt=f"Erectile Dysfunction Severity: {ed_severity}", ln=True)
pdf.cell(200, 10, txt=f"Predicted Recurrence Probability: {recurrence_prob:.2%}", ln=True)

pdf_buffer = io.BytesIO()
pdf.output(pdf_buffer)
pdf_buffer.seek(0)

st.download_button(
    label="📄 Download Summary as PDF",
    data=pdf_buffer,
    file_name="recurrence_summary.pdf",
    mime="application/pdf"
)
