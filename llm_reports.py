import os
import pandas as pd
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import math

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file!")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Load your CSV file
df = pd.read_csv("pdf_summaries.csv")

# Ensure date column is parsed properly
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# Extract month, quarter, and year
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["quarter"] = df["date"].dt.quarter

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

def generate_readable_summary(text, period_label):
    """Ask LLM to create a readable intelligence-style summary paragraph."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a cyber intelligence analyst writing monthly and yearly fraud summaries. "
                "Use natural language and full sentences, not bullet points. "
                "Summarize key trends, recurring fraud types, and emerging threats clearly."
            ),
        },
        {
            "role": "user",
            "content": f"""
Summarize the following text into a professional summary for {period_label}.
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
    combined_summary = generate_readable_summary(" ".join(partial_summaries), f"final summary for {period_label}")
    return combined_summary

def combine_and_summarize(df, groupby_cols, label_func):
    """Combine text by group and generate summaries."""
    reports = []
    for group_vals, group_df in df.groupby(groupby_cols):
        label = label_func(group_vals)
        combined_text = " ".join(group_df["summary"].astype(str).tolist())
        if len(combined_text.strip()) < 100:
            continue
        summary = summarize_large_text(combined_text, period_label=label)
        reports.append((label, summary))
    return reports

# --- Generate reports ---
monthly_reports = combine_and_summarize(
    df,
    ["year", "month"],
    lambda g: datetime(g[0], g[1], 1).strftime("%B %Y"),
)

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

# --- Overall summary ---
overall_text = " ".join(df["summary"].astype(str).tolist())
overall_summary = summarize_large_text(overall_text, "the overall 2020–2025 period")

# --- Save all summaries to file ---
with open("fraud_reports.txt", "w", encoding="utf-8") as f:
    f.write("=== Monthly Summaries ===\n\n")
    for label, summary in monthly_reports:
        f.write(f"## {label}\n{summary}\n\n")

    f.write("\n=== Quarterly Summaries ===\n\n")
    for label, summary in quarterly_reports:
        f.write(f"## {label}\n{summary}\n\n")

    f.write("\n=== Yearly Summaries ===\n\n")
    for label, summary in yearly_reports:
        f.write(f"## {label}\n{summary}\n\n")

    f.write("\n=== Overall Report ===\n\n")
    f.write(overall_summary)

print("✅ All summaries generated and saved in fraud_reports.txt")
