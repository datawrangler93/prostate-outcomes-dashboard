import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv("cleaned_prostate_treatment.csv")

st.title("Dashboard")
st.write("A visual summary of prostate cancer treatment outcomes")

# Optional: show raw data
if st.checkbox("Show raw data"):
    st.dataframe(df)

# --- Gleason Score Distribution ---
st.subheader("Distribution of Gleason Scores")
fig1, ax1 = plt.subplots()
sns.countplot(data=df, x='Gleason_Score', order=sorted(df['Gleason_Score'].unique()), ax=ax1)
ax1.set_title("Distribution of Gleason Scores")
ax1.set_ylabel("Patient Count")
st.pyplot(fig1)

# --- Side Effect Severity by Treatment Type ---
st.subheader("Average Side Effect Severity by Treatment")
severity_df = df.groupby('Treatment_Type')[['Urinary_Score', 'ED_Score']].mean().reset_index()
st.dataframe(severity_df)

fig2, ax2 = plt.subplots()
severity_df.plot(kind='bar', x='Treatment_Type', ax=ax2)
ax2.set_title("Avg Urinary & ED Severity by Treatment")
ax2.set_ylabel("Severity (0=None, 2=Severe)")
st.pyplot(fig2)