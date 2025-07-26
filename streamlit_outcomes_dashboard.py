import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("cleaned_prostate_treatment.csv")

st.title("🧬 Prostate Cancer Outcomes Dashboard")
st.markdown("Explore treatment comparisons, PSA trends, side effects, and Gleason score patterns.")

# Sidebar filters
st.sidebar.header("Filters")
treatment_options = st.sidebar.multiselect("Select Treatment Types:", df["Treatment_Type"].unique(), default=df["Treatment_Type"].unique())
filtered_df = df[df["Treatment_Type"].isin(treatment_options)]

# Dataset info
st.subheader("📊 Dataset Overview")
st.write("Shape:", filtered_df.shape)
st.write("Missing values:", filtered_df.isnull().sum())

# PSA Boxplot
st.subheader("📉 PSA at 6 Months by Treatment")
fig1, ax1 = plt.subplots()
sns.boxplot(data=filtered_df, x="Treatment_Type", y="PSA_6_Months", ax=ax1)
plt.xticks(rotation=15)
st.pyplot(fig1)

# Recurrence Rate
st.subheader("🔁 Recurrence Rate by Treatment")
recurrence = filtered_df.groupby("Treatment_Type")["Recurrence_Flag_Binary"].mean() * 100
st.dataframe(recurrence.round(1).rename("Recurrence %"))

# Side Effects
st.subheader("⚠ Average Side Effect Severity (0=None, 2=Severe)")
side_effects = filtered_df.groupby("Treatment_Type")[["Urinary_Score", "ED_Score"]].mean()
st.dataframe(side_effects.round(2))

# Gleason Score Distribution
st.subheader("📊 Gleason Score Distribution")
fig2, ax2 = plt.subplots()
sns.countplot(data=filtered_df, x="Gleason_Score", order=sorted(filtered_df["Gleason_Score"].unique()), ax=ax2)
plt.title("Distribution of Gleason Scores")
st.pyplot(fig2)

# Gleason Score by Treatment (stacked bar)
st.subheader("📊 Gleason Score by Treatment Type")
gleason_counts = pd.crosstab(filtered_df["Treatment_Type"], filtered_df["Gleason_Score"])
fig3, ax3 = plt.subplots()
gleason_counts.plot(kind="bar", stacked=True, ax=ax3)
plt.title("Gleason Score Distribution by Treatment")
plt.xlabel("Treatment Type")
plt.ylabel("Patient Count")
plt.xticks(rotation=15)
st.pyplot(fig3)

st.markdown("---")
st.caption("Demo dashboard created with Streamlit · Data is simulated for educational purposes only.")