import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load data
df = pd.read_csv("cleaned_prostate_treatment.csv")

# Dummy model (for demo only)
model = LogisticRegression()
X_dummy = df[['Gleason_Score', 'PSA_Baseline', 'PSA_6mo', 'Urinary_Score', 'ED_Score']].fillna(0)
y_dummy = np.random.randint(0, 2, len(X_dummy))
model.fit(X_dummy, y_dummy)

# Layout tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Recurrence Predictor"])

# Dashboard Tab
with tab1:
    st.title("Dashboard")
    st.write("A visual summary of prostate cancer treatment outcomes")

    if st.checkbox("Show raw data"):
        st.dataframe(df)

    st.subheader("Distribution of Gleason Scores")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='Gleason_Score', order=sorted(df['Gleason_Score'].unique()), ax=ax1)
    ax1.set_title("Distribution of Gleason Scores")
    ax1.set_ylabel("Patient Count")
    st.pyplot(fig1)

    st.subheader("Average Side Effect Severity by Treatment")
    severity_df = df.groupby('Treatment_Type')[['Urinary_Score', 'ED_Score']].mean().reset_index()
    st.dataframe(severity_df)

    fig2, ax2 = plt.subplots()
    severity_df.plot(kind='bar', x='Treatment_Type', ax=ax2)
    ax2.set_title("Avg Urinary & ED Severity by Treatment")
    ax2.set_ylabel("Severity (0=None, 2=Severe)")
    st.pyplot(fig2)

# Predictor Tab
with tab2:
    st.title("🧠 Recurrence Risk Predictor")
    st.write("Input patient features to estimate recurrence probability")

    gleason_score = st.slider("Gleason Score (6–10)", 6, 10, step=1)
    psa_baseline = st.number_input("PSA Baseline", value=10.0)
    psa_6mo = st.number_input("PSA at 6 Months", value=1.0)

    urinary_severity = st.selectbox("Urinary Incontinence Severity", ['None', 'Mild', 'Moderate', 'Severe'])
    ed_severity = st.selectbox("Erectile Dysfunction Severity", ['None', 'Mild', 'Moderate', 'Severe'])

    severity_map = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
    urinary_score = severity_map[urinary_severity]
    ed_score = severity_map[ed_severity]

    if st.button("Predict Recurrence Probability"):
        input_features = pd.DataFrame([[
            gleason_score, psa_baseline, psa_6mo, urinary_score, ed_score
        ]], columns=['Gleason_Score', 'PSA_Baseline', 'PSA_6mo', 'Urinary_Score', 'ED_Score'])

        prob = model.predict_proba(input_features)[0][1]
        st.success(f"Estimated probability of recurrence: {prob * 100:.2f}%")
        
