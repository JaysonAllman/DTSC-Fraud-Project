# -----------------------------
# Streamlit Fraud Trend Dashboard
# -----------------------------
import pandas as pd
import streamlit as st
from collections import Counter
import importlib.util
from pathlib import Path

# Page setup
st.set_page_config(page_title="Fraud Trend Dashboard", page_icon="🔎", layout="centered")

# Title
st.markdown(
    "<h1 style='text-align:center;color:#0A2E5C;'>🔎 Fraud Trend Dashboard</h1>",
    unsafe_allow_html=True,
)

# Load paths
csv_path = Path("pdf_summaries.csv")
keywords_path = Path("crime_keywords.py")

# Check files exist
if not csv_path.exists():
    st.error("Missing pdf_summaries.csv — run scraping first.")
    st.stop()

if not keywords_path.exists():
    st.error("Missing crime_keywords.py — add it to this folder.")
    st.stop()

# Load CSV
df = pd.read_csv(csv_path)

# Load keyword trends
spec = importlib.util.spec_from_file_location("crime_keywords", keywords_path)
kw_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kw_mod)

TRENDS = getattr(kw_mod, "TRENDS", {})
phrases = [p for v in TRENDS.values() for p in v]

# Extract text and count phrases
text_cols = [c for c in df.columns if df[c].dtype == object]
text_data = " ".join(" ".join(str(df[c].fillna('').tolist())) for c in text_cols).lower()

phrase_counts = Counter()
for phrase in phrases:
    phrase_counts[phrase] = text_data.count(phrase.lower())

# Top 5 keywords
top5 = pd.DataFrame(phrase_counts.most_common(5), columns=["Keyword/Phrase", "Mentions"])

# Top 3 trends
trend_totals = {trend: sum(phrase_counts[p] for p in words) for trend, words in TRENDS.items()}
top3 = pd.DataFrame(sorted(trend_totals.items(), key=lambda x: x[1], reverse=True)[:3],
                    columns=["Trend", "Mentions"])

# Dark blue theme styling
st.markdown(
    """
    <style>
    body { background-color: #0A2E5C; color: white; }
    .stDataFrame { background-color: #0A2E5C !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Show charts
st.subheader("Top 5 Keywords / Phrases")
st.bar_chart(top5.set_index("Keyword/Phrase"), color="#0A2E5C")

st.subheader("Top 3 Fraud Trends")
st.bar_chart(top3.set_index("Trend"), color="#0A2E5C")

# Footer
st.markdown("<p style='text-align:center;color:gray;'>© 2025 Fraud Project</p>", unsafe_allow_html=True)
