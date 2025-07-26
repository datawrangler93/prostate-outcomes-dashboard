import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("Prostate Treatment Outcomes Dashboard")

# Load data
df = pd.read_csv("cleaned_prostate_treatment.csv")

# Show raw data
if st.checkbox("Show raw data"):
    st.write(df.head())

# Gleason Score distribution
st.subheader("Distribution of Gleason Scores")
fig1, ax1 = plt.subplots()
sns.countplot(data=df, x='Gleason_Score', order=sorted(df['Gleason_Score'].unique()), ax=ax1)
st.pyplot(fig1)

# Side effect severity by treatment type
st.subheader("Average Side Effect Severity by Treatment")
avg_scores = df.groupby('Treatment_Type')[['Urinary_Score', 'ED_Score']].mean()
st.write(avg_scores)