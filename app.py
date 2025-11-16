import streamlit as st
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt  # for pie chart

# -------------------------
# Load secrets from .env
# -------------------------
load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

# Connect to Supabase
supabase = create_client(url, key)

# -------------------------
# Get data from Supabase
# -------------------------
@st.cache_data
def load_data():
    data = supabase.table("fraud_keywords").select("*").execute()
    df = pd.DataFrame(data.data)
    return df

df = load_data()

# -------------------------
# Streamlit Page Setup
# -------------------------
st.set_page_config(page_title="Fraud Dashboard", layout="centered")

# White background + dark blue text
st.markdown("""
    <style>
        body, .stApp {
            background-color: white !important;
            color: #0a1a3c !important;
        }
        h1, h2, h3, h4, h5, h6, div, p, span, label {
            color: #0a1a3c !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title + bold subtitle
st.title("📊 Fraud Keyword Analysis Dashboard")
st.markdown("<h3><b>DTSC Project Team 2</b></h3>", unsafe_allow_html=True)

# -------------------------
# YEAR DROPDOWN FILTER
# -------------------------
st.subheader("🔽 Filter by Year")

years = ["All Years"] + sorted(df["year"].unique().tolist())
selected_year = st.selectbox("Choose a year:", years)

if selected_year != "All Years":
    filtered_df = df[df["year"] == selected_year]
else:
    filtered_df = df.copy()

# -------------------------
# SHOW FILTERED RAW DATA
# -------------------------
st.subheader(" Filtered Keyword Data")
st.dataframe(filtered_df)

# -------------------------
# DOWNLOAD CSV BUTTON
# -------------------------
st.subheader("⬇️ Download Filtered Data")
st.write("If you can see this text, the download button is directly below:")

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv_data,
    file_name="filtered_fraud_keywords.csv",
    mime="text/csv"
)

# -------------------------
# TOP 5 KEYWORDS (Filtered)
# -------------------------
st.subheader(" Top 5 Keywords (Filtered View)")

if not filtered_df.empty:
    top_keywords = (
        filtered_df.groupby("keyword")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    st.bar_chart(top_keywords)

    # -------------------------
    # PIE CHART FOR TOP 5 KEYWORDS
    # -------------------------
    st.subheader(" Keyword Share (Top 5, Filtered)")

    fig, ax = plt.subplots()
    ax.pie(
        top_keywords.values,
        labels=top_keywords.index,
        autopct="%1.1f%%"
    )
    ax.set_title("Top 5 Keyword Share")

    st.pyplot(fig)
else:
    st.info("No data available for this filter.")

# -------------------------
# TOP 3 TRENDS OVER TIME
# -------------------------
st.subheader(" Top 3 Fraud Trends Over Time (All Data)")

trends = (
    df.groupby("year")["count"]
    .sum()
    .sort_values(ascending=False)
    .head(3)
)

st.line_chart(trends)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")

