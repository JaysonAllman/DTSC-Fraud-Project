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
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from openai import OpenAI

# -----------------------------
# 1️⃣ Load environment variables
# -----------------------------
load_dotenv()  # This reads your .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found. Please check your .env file.")
    st.stop()  # stop the app if key is missing

# -----------------------------
# 2️⃣ Initialize OpenAI client
# -----------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# 3️⃣ Streamlit UI
# -----------------------------
st.title("OpenAI Test App")

prompt = st.text_input("Enter a prompt:")

if st.button("Generate"):
    if prompt:
        # Example usage
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        st.write(response.choices[0].message.content)
    else:
        st.warning("Please enter a prompt first.")

# ---------------------------
# Configuration
# ---------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing SUPABASE_URL or SUPABASE_KEY in environment. Put them in your .env file.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------
# Helpers
# ---------------------------
def parse_json_field(field: Any) -> Dict:
    if field is None:
        return {}
    if isinstance(field, dict):
        return field
    if isinstance(field, str):
        try:
            return json.loads(field)
        except Exception:
            try:
                return json.loads(field.replace("'", '"'))
            except Exception:
                return {}
    return {}

def fetch_pdf_summaries() -> pd.DataFrame:
    res = supabase.table("pdf_summaries").select("*").execute()
    if res.data is None:
        st.error(f"Error fetching pdf_summaries. Status code: {res.status_code}")
        return pd.DataFrame()
    df = pd.DataFrame(res.data)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["fraud_type_counts_parsed"] = df["fraud_type_counts"].apply(parse_json_field)
    df["keyword_counts_parsed"] = df["keyword_counts"].apply(parse_json_field)
    return df

def fetch_fraud_reports() -> pd.DataFrame:
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
    totals = {}
    for fraud_group, inner in (keyword_json or {}).items():
        if not isinstance(inner, dict):
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
    rows = []
    for _, r in df.iterrows():
        dt = r["date"]
        if agg == "month":
            label = dt.strftime("%Y-%m")
        elif agg == "quarter":
            label = f"{dt.year}-Q{r['quarter']}"
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
    grouped = ts_df.groupby("period").sum().reset_index()
    totals_all = grouped.drop(columns=["period"]).sum().sort_values(ascending=False)
    top_types = list(totals_all.head(top_n).index)
    cols = ["period"] + top_types
    result = grouped[cols].sort_values("period")
    for c in top_types:
        result[c] = result[c].astype(int)
    return result

def get_top_keywords(keyword_totals: Dict[str,int], top_n: int = 5) -> List[Tuple[str,int]]:
    items = sorted(keyword_totals.items(), key=lambda x: x[1], reverse=True)
    return items[:top_n]

def get_llm_report_for_period(reports_df: pd.DataFrame, selection_mode: str, selection_value: Any) -> str:
    if reports_df.empty:
        return ""
    if selection_mode == "All":
        label = "Overall 2020–2025"
    elif selection_mode == "Year":
        label = str(selection_value)
    elif selection_mode == "Quarter":
        year, q = selection_value
        label = f"Q{q} {year}"
    elif selection_mode == "MonthRange":
        start, end = selection_value
        if start.year == end.year and start.month == end.month:
            label = start.strftime("%B %Y")
        else:
            label = f"{start.strftime('%b %Y')} - {end.strftime('%b %Y')}"
    else:
        label = ""
    matches = reports_df[reports_df["period"].astype(str).str.strip().str.lower() == label.strip().lower()]
    if not matches.empty:
        return matches.iloc[0]["summary"]
    matches = reports_df[reports_df["period"].astype(str).str.lower().str.contains(label.strip().lower(), na=False)]
    if not matches.empty:
        return matches.iloc[0]["summary"]
    if selection_mode in ("Quarter", "MonthRange"):
        yr = selection_value[0] if selection_mode=="Quarter" else selection_value[0].year
        matches = reports_df[reports_df["period"].astype(str).str.strip() == str(yr)]
        if not matches.empty:
            return matches.iloc[0]["summary"]
    return ""

# ---------------------------
# Fraud Score, Clustering, Fraud Weight
# ---------------------------
def compute_fraud_score(row, kw_weight=1.0, ft_weight=1.5):
    kw_count = sum(flatten_keyword_counts(row["keyword_counts_parsed"]).values())
    ft_count = sum(row["fraud_type_counts_parsed"].values() if row["fraud_type_counts_parsed"] else [])
    return kw_weight * kw_count + ft_weight * ft_count

def compute_clusters(df, n_clusters=5):
    if "embedding" not in df.columns or df.empty:
        return df
    embeddings = np.array(df["embedding"].tolist())
    if len(embeddings) < n_clusters:
        n_clusters = max(1, len(embeddings)//2)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(embeddings)
    return df

def compute_fraud_weight(df):
    if "fraud_score" not in df.columns:
        df["fraud_score"] = df.apply(compute_fraud_score, axis=1)
    cluster_avg = df.groupby("cluster")["fraud_score"].transform("mean")
    df["fraud_weight_raw"] = df["fraud_score"] * cluster_avg
    scaler = MinMaxScaler()
    df["fraud_weight"] = scaler.fit_transform(df[["fraud_weight_raw"]])
    return df

def risk_level(score):
    if score < 0.33:
        return "Low"
    elif score < 0.66:
        return "Medium"
    return "High"

def semantic_search(df, query, top_k=5):
    resp = openai_client.embeddings.create(model="text-embedding-3-small", input=query)
    query_emb = np.array(resp.data[0].embedding)
    df["similarity"] = df["embedding"].apply(lambda e: cosine_similarity([query_emb], [e])[0][0])
    results = df.sort_values("similarity", ascending=False).head(top_k)
    return results

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Fraud Trends Dashboard", layout="wide")
st.title("Fraud Trends Dashboard")

@st.cache_data(ttl=300)
def load_data():
    pdf_df = fetch_pdf_summaries()
    reports_df = fetch_fraud_reports()
    return pdf_df, reports_df

pdf_df, reports_df = load_data()
if pdf_df is None or pdf_df.empty:
    st.warning("No pdf_summaries rows found in Supabase.")
    st.stop()

# Sidebar filters
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

# Filter pdf rows
filtered = timeframe_filter(pdf_df, selection_mode if selection_mode!="All" else "All", selection_value)

st.sidebar.markdown(f"**PDF rows in selection:** {len(filtered)}")

# Aggregate totals
fraud_totals = aggregate_fraud_type_counts(filtered)
keyword_totals = aggregate_keyword_counts(filtered)
ts_df = timeseries_by_period(filtered, agg=agg_resolution, top_n=topn)

# Tabs: Dashboard vs Semantic Search
tab1, tab2 = st.tabs(["Dashboard", "Semantic Search"])

with tab1:
    # ---------------------------
    # Fraud trends chart
    # ---------------------------
    col1, col2 = st.columns([3,1])
    with col1:
        st.subheader("Fraud type trends")
        if ts_df.empty:
            st.info("Not enough data to build a trend for this selection.")
        else:
            melted = ts_df.melt(id_vars=["period"], var_name="fraud_type", value_name="count")
            def period_label_to_dt(row):
                p = row['period']
                try:
                    if "-" in p and len(p.split("-")[1]) == 2:
                        y, m = p.split("-")
                        return datetime(int(y), int(m), 1)
                    if "-Q" in p:
                        y, q = p.split("-Q")
                        y = int(y)
                        q = int(q)
                        match_dates = filtered[(filtered['year'] == y) & (filtered['quarter'] == q)]['date']
                        return match_dates.min().replace(day=1) if not match_dates.empty else datetime(y, (q-1)*3+1,1)
                    return datetime(int(p),1,1)
                except Exception:
                    return datetime(1970,1,1)
            melted["period_dt"] = melted.apply(period_label_to_dt, axis=1)
            chart = alt.Chart(melted).mark_line(point=True).encode(
                x=alt.X("period_dt:T", title="Period", axis=alt.Axis(labelExpr='timeFormat(datum.value, "%Y-%m")')),
                y=alt.Y("count:Q", title="Total mentions"),
                color=alt.Color("fraud_type:N", title="Fraud type"),
                tooltip=[alt.Tooltip("period_dt:T", title="Period"), alt.Tooltip("fraud_type:N", title="Fraud Type"), alt.Tooltip("count:Q", title="Count")]
            ).properties(width=900, height=420)
            st.altair_chart(chart, use_container_width=True)
    with col2:
        st.subheader(f"Top {topk} Keywords")
        top_keywords = get_top_keywords(keyword_totals, topk)
        if not top_keywords:
            st.write("No keyword data available for the selected period.")
        else:
            for i, (kw, cnt) in enumerate(top_keywords, start=1):
                st.write(f"**{i}. {kw}** — {cnt:,}")

    # LLM narrative
    st.markdown("---")
    st.subheader("AI Narrative")
    llm_text = get_llm_report_for_period(reports_df, "All" if selection_mode=="All" else selection_mode, selection_value)
    if not llm_text:
        st.info("No LLM narrative found for the selected period.")
    else:
        st.write(llm_text)

    # Fraud scoring, clustering, fraud weight
    st.markdown("---")
    st.subheader("Fraud Scoring, Clustering & Fraud Weight")
    filtered_scoring = filtered.copy()
    filtered_scoring = compute_clusters(filtered_scoring)
    filtered_scoring = compute_fraud_weight(filtered_scoring)
    filtered_scoring["risk_level"] = filtered_scoring["fraud_weight"].apply(risk_level)
    st.dataframe(filtered_scoring[["title","fraud_score","cluster","fraud_weight","risk_level"]].sort_values("fraud_weight", ascending=False))

    # Cluster distribution
    st.subheader("Cluster distribution")
    cluster_counts = filtered_scoring.groupby("cluster").size().reset_index(name="count")
    st.bar_chart(cluster_counts.set_index("cluster"))

with tab2:
    st.subheader("Semantic Search")
    query = st.text_input("Enter search query:")
    top_k = st.slider("Top results to show", 1, 10, 5)
    if query:
        search_results = semantic_search(pdf_df, query, top_k=top_k)
        for _, row in search_results.iterrows():
            snippet = (row["text"][:200] + "...") if row["text"] else ""
            weight = row["fraud_weight"] if "fraud_weight" in row else 0
            st.markdown(f"**{row['title']}** — Risk: {risk_level(weight)}")
            st.write(snippet)
            st.markdown("---")

# Footer / tips
st.markdown("""
**Definitions**
- Advanced Fee Fraud: An individual pays money to someone in anticipation of receiving something of greater value in return, but instead, receives significantly less than expected or nothing.
- Business Email Compromise (BEC): BEC is a scam targeting businesses or individuals working with suppliers and/or businesses regularly performing wire transfer payments. These sophisticated scams are carried out by fraudsters by compromising email accounts and other forms of communication such as phone numbers and virtual meeting applications, through social engineering or computer intrusion techniques to conduct unauthorized transfer of funds.
- Botnet: A botnet is a group of two or more computers controlled and updated remotely for an illegal purchase such as a Distributed Denial of Service or Telephony Denial of Service attack or other nefarious activity.
- Confidence/Romance Fraud: An individual believes they are in a relationship (family, friendly, or romantic) and are tricked into sending money, personal and financial information, or items of value to the perpetrator or to launder money or items to assist the perpetrator. This includes the Grandparent’s Scheme and any scheme in which the perpetrator preys on the targeted individual’s “heartstrings.”
- Credit Card Fraud/Check Fraud: Credit card fraud is a wide-ranging term for theft and fraud committed using a credit card or any similar payment mechanism (ACH, EFT, recurring charge, etc.) as a fraudulent source of funds in a transaction.
- Crimes Against Children: Anything related to the exploitation of children, including child abuse. 
- Data Breach: A data breach in the cyber context is the use of a computer intrusion to acquire confidential or secured information. This does not include computer intrusions targeting personally owned computers, systems, devices, or personal accounts such as social media or financial accounts.
- Employment Fraud: An individual believes they are legitimately employed and loses money, or launders money/items during their employment.
- Extortion: Unlawful extraction of money or property through intimidation or undue exercise of authority. It may include threats of physical harm, criminal prosecution, or public exposure. 
- Government Impersonation: A government official is impersonated to collect or extort money. 
- Harassment/Stalking: Repeated words, conduct, and/or action that serve no legitimate purpose and are directed at a specific person to annoy, alarm, or distress that person. Engaging in a course of conduct directed at a specific person that would cause a reasonable person to fear for his/her safety or the safety of others or suffer substantial emotional distress.
- Identity Theft: Someone wrongfully obtains and uses personally identifiable information in some way that involves fraud or deception, typically for economic gain. 
- Investment Fraud: Deceptive practice that induces investors to make purchases based on false information. These scams usually offer those targeted large returns with minimal risk. (Retirement, 401K, Ponzi, Pyramid, etc.).
- Intellectual Property Rights (IPR)/Copyright and Counterfeit: The illegal theft and use of others’ ideas, inventions, and creative expressions – what’s called intellectual property – everything from trade secrets and proprietary products and parts to movies, music, and software. 
- Lottery/Sweepstakes/Inheritance Fraud: An individual is contacted about winning a lottery or sweepstakes they never entered, or to collect on an inheritance from an unknown relative.
- Malware: Software or code intended to damage, disable, or capable of copying itself onto a computer and/or computer systems to have a detrimental effect or destroy data. 
- Non-Payment/Non-Delivery Fraud: Goods or services are shipped, and payment is never rendered (nonpayment). Payment is sent, and goods or services are never received, or are of lesser quality (nondelivery).
- Overpayment: An individual is sent a payment/commission and is instructed to keep a portion of the payment and send the remainder to another individual or business.
- Personal Data Breach: A leak/spill of personal data which is released from a secure location to an untrusted environment. Also, a security incident in which an individual’s sensitive, protected, or confidential data is copied, transmitted, viewed, stolen, or used by an unauthorized individual.
- Phishing/Spoofing: The use of unsolicited email, text messages, and telephone calls purportedly from a legitimate company requesting personal, financial, and/or login credentials. 
- Ransomware: A type of malicious software designed to block access to a computer system until money is paid. 
- Real Estate Fraud: Loss of funds from a real estate investment or fraud involving rental or timeshare property.
- SIM Swap: The use of unsophisticated social engineering techniques against mobile service providers to transfer a victim’s phone service to a mobile device in the criminal’s possession.
- Tech Support Fraud: Subject posing as technical or customer support/service.
- Threats of Violence: An expression of an intention to inflict pain, injury, self-harm, or death not in the context of extortion. 
Information provided by: Internet Crime Complaint Center (IC3). “Internet Crime Complaint Center(IC3) | Annual Crime Report 2024.” Www.ic3.Gov, 2024, www.ic3.gov/.
""")
