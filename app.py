# streamlit_dashboard.py
import os
import json
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Configuration
# ---------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing SUPABASE_URL or SUPABASE_KEY in environment. Put them in your .env file.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------
# Helpers
# ---------------------------
def parse_json_field(field: Any) -> Dict:
    """Supabase may return a dict already or a JSON string; normalize to dict."""
    if field is None:
        return {}
    if isinstance(field, dict):
        return field
    if isinstance(field, str):
        try:
            return json.loads(field)
        except Exception:
            # try single quotes fallback
            try:
                return json.loads(field.replace("'", '"'))
            except Exception:
                return {}
    return {}

def fetch_pdf_summaries() -> pd.DataFrame:
    """Fetch all rows from pdf_summaries table and convert to DataFrame."""
    res = supabase.table("pdf_summaries").select("*").execute()

    if res.data is None:
        st.error(f"Error fetching pdf_summaries. Status code: {res.status_code}")
        return pd.DataFrame()

    df = pd.DataFrame(res.data)
    if df.empty:
        return df

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # Normalize JSON fields
    df["fraud_type_counts_parsed"] = df["fraud_type_counts"].apply(parse_json_field)
    df["keyword_counts_parsed"] = df["keyword_counts"].apply(parse_json_field)

    return df

def fetch_fraud_reports() -> pd.DataFrame:
    """Fetch LLM reports (fraud_reports table)."""
    res = supabase.table("fraud_reports").select("*").execute()

    if res.data is None:
        st.error(f"Error fetching fraud_reports. Status code: {res.status_code}")
        return pd.DataFrame()

    df = pd.DataFrame(res.data)
    if df.empty:
        return df

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df

def flatten_keyword_counts(keyword_json: Dict) -> Dict[str, int]:
    """
    keyword_json is expected like:
    { "Malware": {"malware":5, "trojan":1}, "Phishing/Spoofing": {"phishing":2} }
    return combined totals across all fraud groups: {"malware":5, "trojan":1, "phishing":2}
    """
    totals = {}
    for fraud_group, inner in (keyword_json or {}).items():
        if not isinstance(inner, dict):
            # If stored as integer or non-dict, ignore or try to coerce
            continue
        for term, cnt in inner.items():
            try:
                c = int(cnt)
            except Exception:
                c = 0
            totals[term] = totals.get(term, 0) + c
    return totals

def aggregate_fraud_type_counts(rows: pd.DataFrame) -> Dict[str, int]:
    totals = {}
    for d in rows["fraud_type_counts_parsed"]:
        for k, v in (d or {}).items():
            try:
                c = int(v)
            except Exception:
                c = 0
            totals[k] = totals.get(k, 0) + c
    return totals

def aggregate_keyword_counts(rows: pd.DataFrame) -> Dict[str, int]:
    totals = {}
    for d in rows["keyword_counts_parsed"]:
        flattened = flatten_keyword_counts(d)
        for term, count in flattened.items():
            totals[term] = totals.get(term, 0) + int(count)
    return totals

def timeframe_filter(df: pd.DataFrame, selection_mode: str, selection_value: Any) -> pd.DataFrame:
    """
    selection_mode: one of "All", "Year", "Quarter", "MonthRange"
    selection_value:
      - "All": ignored
      - "Year": int year
      - "Quarter": tuple (year, quarter) e.g. (2020, 3)
      - "MonthRange": tuple (start_date, end_date) as datetime
    """
    if selection_mode == "All":
        return df.copy()
    if selection_mode == "Year":
        year = selection_value
        return df[df["year"] == int(year)].copy()
    if selection_mode == "Quarter":
        year, q = selection_value
        return df[(df["year"] == int(year)) & (df["quarter"] == int(q))].copy()
    if selection_mode == "MonthRange":
        start, end = selection_value
        return df[(df["date"] >= start) & (df["date"] <= end)].copy()
    return df.copy()

def timeseries_by_period(df: pd.DataFrame, agg: str="quarter", top_n:int=5) -> pd.DataFrame:
    """
    Build a timeseries DataFrame of counts grouped by period resolution (month, quarter, year)
    We will sum fraud_type_counts across rows for each period and return DataFrame with
    columns: period_label, fraud_type1, fraud_type2, ...
    """
    rows = []
    # expand per-row fraud_type_counts to per-row dict
    for _, r in df.iterrows():
    dt = r["date"]
    quarter = int(r['quarter']) if pd.notnull(r['quarter']) else 1  # fallback to Q1 if missing
    if agg == "month":
        label = dt.strftime("%Y-%m")
    elif agg == "quarter":
        label = f"{dt.year}-Q{quarter}"
    else:
        label = str(dt.year)
    totals = {}
    for k, v in (r["fraud_type_counts_parsed"] or {}).items():
        try:
            c = int(v)
        except Exception:
            c = 0
        totals[k] = totals.get(k, 0) + c
    rows.append({"period": label, **totals})

    if not rows:
        return pd.DataFrame()

    ts_df = pd.DataFrame(rows).fillna(0)
    # group by period label and sum
    grouped = ts_df.groupby("period").sum().reset_index()
    # Find global top fraud types (top_n) across the whole period
    if grouped.shape[0] == 0:
        return grouped
    totals_all = grouped.drop(columns=["period"]).sum().sort_values(ascending=False)
    top_types = list(totals_all.head(top_n).index)
    # restrict grouped to period + top types
    cols = ["period"] + top_types
    result = grouped[cols].sort_values("period")
    # ensure numeric
    for c in top_types:
        result[c] = result[c].astype(int)
    return result

def get_top_keywords(keyword_totals: Dict[str,int], top_n: int = 5) -> List[Tuple[str,int]]:
    items = sorted(keyword_totals.items(), key=lambda x: x[1], reverse=True)
    return items[:top_n]

def get_llm_report_for_period(reports_df: pd.DataFrame, selection_mode: str, selection_value: Any) -> str:
    """Find best matching LLM report by period label. Returns text or empty string."""
    if reports_df.empty:
        return ""
    # Normalize different ways user might request period labels
    if selection_mode == "All":
        label = "Overall 2020–2025"
    elif selection_mode == "Year":
        label = str(selection_value)
    elif selection_mode == "Quarter":
        year, q = selection_value
        label = f"Q{q} {year}"
    elif selection_mode == "MonthRange":
        start, end = selection_value
        # try to find a monthly named report if one exists
        # prefer "Month Year" format for single-month ranges
        if start.year == end.year and start.month == end.month:
            label = start.strftime("%B %Y")
        else:
            # fallback: use "start - end"
            label = f"{start.strftime('%b %Y')} - {end.strftime('%b %Y')}"
    else:
        label = ""

    # Attempt to match the 'period' column exactly
    matches = reports_df[reports_df["period"].astype(str).str.strip().str.lower() == label.strip().lower()]
    if not matches.empty:
        return matches.iloc[0]["summary"]

    # fallback: partial match
    matches = reports_df[reports_df["period"].astype(str).str.lower().str.contains(label.strip().lower(), na=False)]
    if not matches.empty:
        return matches.iloc[0]["summary"]

    # If still not found, try to return the closest year-level report (for month/quarter)
    if selection_mode in ("Quarter", "MonthRange"):
        yr = selection_value[0] if selection_mode=="Quarter" else selection_value[0].year
        matches = reports_df[reports_df["period"].astype(str).str.strip() == str(yr)]
        if not matches.empty:
            return matches.iloc[0]["summary"]

    return ""

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Fraud Trends Dashboard", layout="wide")
st.title("Fraud Trends Dashboard (Supabase → Streamlit)")

# Load data once (caching)
@st.cache_data(ttl=300)
def load_data():
    pdf_df = fetch_pdf_summaries()
    reports_df = fetch_fraud_reports()
    return pdf_df, reports_df

pdf_df, reports_df = load_data()
if pdf_df is None or pdf_df.empty:
    st.warning("No pdf_summaries rows found in Supabase.")
    st.stop()

# Sidebar controls
st.sidebar.header("Filters & Controls")
years = sorted(pdf_df["year"].unique().tolist())
min_year, max_year = min(years), max(years)
selection_mode = st.sidebar.selectbox("Select time window type", ["All", "Year", "Quarter", "Custom Range (months)"])

selection_value = None
if selection_mode == "Year":
    sel_year = st.sidebar.selectbox("Year", years, index=len(years)-1)
    selection_value = int(sel_year)
elif selection_mode == "Quarter":
    sel_year = st.sidebar.selectbox("Year", years, index=len(years)-1)
    sel_q = st.sidebar.selectbox("Quarter", [1,2,3,4], index=3)
    selection_value = (int(sel_year), int(sel_q))
elif selection_mode == "Custom Range (months)":
    start = st.sidebar.date_input("Start date", value=datetime(min_year,1,1).date())
    end = st.sidebar.date_input("End date", value=datetime(max_year,12,31).date())
    if start > end:
        st.sidebar.error("Start date must be <= end date")
    selection_value = (pd.to_datetime(start), pd.to_datetime(end))
else:
    selection_value = None

agg_resolution = st.sidebar.selectbox("Trend resolution (for line chart)", ["year", "quarter", "month"], index=1)
topn = st.sidebar.slider("Top N fraud types to show", 1, 8, 5)
topk = st.sidebar.slider("Top N keywords to list", 1, 10, 5)

# Filter pdf rows by user selection
filtered = timeframe_filter(pdf_df, selection_mode if selection_mode!="All" else "All", selection_value)

st.sidebar.markdown(f"**PDF rows in selection:** {len(filtered)}")

# Aggregate
fraud_totals = aggregate_fraud_type_counts(filtered)
keyword_totals = aggregate_keyword_counts(filtered)

# Timeseries
ts_df = timeseries_by_period(filtered, agg=agg_resolution, top_n=topn)

# Layout: top-left chart, top-right top5 keywords, bottom LLM report
col1, col2 = st.columns([3,1])

with col1:
    st.subheader("Fraud type trends")
    if ts_df.empty:
        st.info("Not enough data to build a trend for this selection.")
    else:
        # melt for altair
        melted = ts_df.melt(id_vars=["period"], var_name="fraud_type", value_name="count")
        # ensure ordering by date: try parse period into sortable ordering
        def period_sort_key(p):
            # p formats: YYYY-Qn, YYYY-MM, or YYYY
            try:
                if "-Q" in p:
                    y, q = p.split("-Q")
                    m = (int(q)-1)*3 + 1
                    return datetime(int(y), m, 1)
                if "-" in p and len(p.split("-")[1])==2:
                    # YYYY-MM
                    y,m = p.split("-")
                    return datetime(int(y), int(m), 1)
                return datetime(int(p), 1, 1)
            except Exception:
                return datetime(1970,1,1)
        melted["period_dt"] = melted["period"].apply(period_sort_key)
        chart = alt.Chart(melted).mark_line(point=True).encode(
            x=alt.X("period_dt:T", title="Period"),
            y=alt.Y("count:Q", title="Total mentions"),
            color=alt.Color("fraud_type:N", title="Fraud type"),
            tooltip=["period:T", "fraud_type:N", "count:Q"]
        ).properties(width=900, height=420)
        st.altair_chart(chart, use_container_width=True)

with col2:
    st.subheader(f"Top {topk} Keywords (selected period)")
    top_keywords = get_top_keywords(keyword_totals, top_k:=topk)
    if not top_keywords:
        st.write("No keyword data available for the selected period.")
    else:
        for i, (kw, cnt) in enumerate(top_keywords, start=1):
            st.write(f"**{i}. {kw}** — {cnt:,}")

# LLM report / narrative
st.markdown("---")
st.subheader("AI Narrative (from fraud_reports table)")
llm_text = get_llm_report_for_period(reports_df, "All" if selection_mode=="All" else selection_mode, selection_value)
if not llm_text:
    st.info("No LLM narrative found for the selected period. You can generate one and store it in the fraud_reports table.")
else:
    st.write(llm_text)

# Small summary area with totals (fraud types)
st.markdown("---")
st.subheader("Aggregate totals for selected period")
if not fraud_totals:
    st.write("No fraud type totals found for this selection.")
else:
    df_totals = pd.DataFrame(sorted(fraud_totals.items(), key=lambda x: x[1], reverse=True), columns=["fraud_type","total"])
    st.dataframe(df_totals)

# Footer / tips
st.markdown("""
**Notes & tips**
- The `pdf_summaries` table should contain JSONB `fraud_type_counts` and `keyword_counts` where `keyword_counts` is nested dictionaries per fraud type (e.g., `{"Malware":{"malware":5,"trojan":1}, "Phishing":{"phishing":2}}`).
- If you want to display monthly charts across wide year ranges, choose `month` resolution; for compact year-to-year comparisons, use `year`.
- To add or update LLM narratives, insert rows into `fraud_reports(period, summary)`.
""")
