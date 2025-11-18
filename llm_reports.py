import os
import pandas as pd
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file!")

client = OpenAI(api_key=api_key)

# -------------------------------
# Load CSV
# -------------------------------
df = pd.read_csv("pdf_summaries_supabase.csv")

# Ensure date column is parsed properly
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# Extract quarter and year
df["quarter"] = df["date"].dt.quarter
df["year"] = df["date"].dt.year

# -------------------------------
# Text chunking
# -------------------------------
def chunk_text(text, max_chars=15000):
    """Split text into smaller chunks to avoid exceeding LLM limits."""
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""
    for sent in sentences:
        if len(current_chunk) + len(sent) + 1 > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = sent + ". "
        else:
            current_chunk += sent + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# -------------------------------
# LLM Summary Generation
# -------------------------------
def generate_readable_summary(text, period_label):
    """Ask LLM to create a readable intelligence-style summary paragraph."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a cyber intelligence analyst writing quarterly, yearly, and overall fraud summaries. "
                "Use full sentences, natural language, not bullet points. "
                "Summarize key trends, recurring fraud types, and emerging threats clearly."
            ),
        },
        {
            "role": "user",
            "content": f"""
Summarize the following text for {period_label}.
Keep it concise (1–2 short paragraphs).

TEXT:
{text}
""",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # fast & cost-efficient
        temperature=0.7,
        max_tokens=400,
        messages=messages,
    )

    return response.choices[0].message.content.strip()

def summarize_large_text(text, period_label):
    """Split text into chunks and summarize each, then combine."""
    chunks = chunk_text(text)
    partial_summaries = []
    for i, chunk in enumerate(chunks):
        partial = generate_readable_summary(chunk, f"{period_label} (part {i+1})")
        partial_summaries.append(partial)
    combined_summary = generate_readable_summary(
        " ".join(partial_summaries), f"final summary for {period_label}"
    )
    return combined_summary

# -------------------------------
# Combine and summarize
# -------------------------------
def combine_and_summarize(df, groupby_cols, label_func):
    """Combine text by group and generate summaries."""
    reports = []
    for group_vals, group_df in df.groupby(groupby_cols):
        label = label_func(group_vals)
        combined_text = " ".join(group_df["summary"].astype(str).tolist())
        if len(combined_text.strip()) < 100:
            continue
        summary = summarize_large_text(combined_text, period_label=label)
        reports.append({"period": label, "summary": summary})
    return reports

# --- Generate reports ---
quarterly_reports = combine_and_summarize(
    df,
    ["year", "quarter"],
    lambda g: f"Q{g[1]} {g[0]}",
)

yearly_reports = combine_and_summarize(
    df,
    ["year"],
    lambda g: str(g[0]),
)

overall_text = " ".join(df["summary"].astype(str).tolist())
overall_summary = summarize_large_text(overall_text, "the overall 2020–2025 period")
overall_reports = [{"period": "Overall 2020–2025", "summary": overall_summary}]

# -------------------------------
# Save all summaries to CSV (Supabase-ready)
# -------------------------------
all_reports = quarterly_reports + yearly_reports + overall_reports
output_df = pd.DataFrame(all_reports)
output_df.to_csv("fraud_reports_supabase.csv", index=False)

print("✅ Quarterly, yearly, and overall summaries generated in fraud_reports_supabase.csv")
