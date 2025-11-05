import os
import re
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import pandas as pd
from datetime import datetime
from urllib.parse import urljoin

from crime_keywords import FRAUD_REGEX 

# -------------------------------
# Helper functions
# -------------------------------

def get_quarter(date):
    """Return quarter number (1-4) from datetime."""
    month = date.month
    if month <= 3:
        return 1
    elif month <= 6:
        return 2
    elif month <= 9:
        return 3
    else:
        return 4

def scrape_pdf_links(page_url):
    """Return list of dicts: [{'title': ..., 'url': ..., 'date': datetime}, ...]"""
    response = requests.get(page_url)
    if response.status_code != 200:
        print(f"Error fetching {page_url}: Status {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    pdf_entries = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag['href']
        if href.lower().endswith(".pdf"):
            title = a_tag.get_text(strip=True)
            full_url = urljoin(page_url, href)

            # Extract date from surrounding text
            date_text = ""
            parent = a_tag.find_parent()
            if parent:
                date_text = parent.get_text(" ", strip=True)

            date_match = re.search(r"\b\w{3},\s+\d{1,2}\s+\w{3}\s+\d{4}\b", date_text)
            if date_match:
                date_obj = datetime.strptime(date_match.group(0), "%a, %d %b %Y")
            else:
                date_obj = None

            pdf_entries.append({
                "title": title,
                "url": full_url,
                "date": date_obj
            })

    return pdf_entries

def download_pdf(url, folder="pdfs"):
    """Download PDF to folder and return local file path."""
    os.makedirs(folder, exist_ok=True)
    filename = os.path.basename(url)
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        return path  # skip download
    r = requests.get(url)
    if r.status_code == 200:
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"Downloaded: {filename}")
        return path
    else:
        print(f"Failed to download {filename}")
        return None

def extract_text(pdf_path):
    """Extract all text from PDF using PyMuPDF."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + " "
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def find_keywords_and_sentences(text):
    """Return a dict of keyword counts and summary sentences containing them."""
    counts = {}
    summary_sentences = []

    sentences = re.split(r'(?<=[.!?])\s+', text)

    for keyword, pattern in FRAUD_REGEX.items():
        match_count = 0
        for sent in sentences:
            if pattern.search(sent):
                match_count += 1
                summary_sentences.append(sent.strip())
        if match_count > 0:
            counts[keyword] = match_count

    # Deduplicate sentences while preserving order
    summary_sentences = list(dict.fromkeys(summary_sentences))

    # Join into one long line, remove newlines
    summary = " ".join(summary_sentences).replace("\n", " ").replace("\r", " ")

    return counts, summary

# -------------------------------
# Main pipeline
# -------------------------------

CSV_FILE = "pdf_summaries.csv"

urls = {
    2020: "https://www.ic3.gov/CSA/2020",
    2021: "https://www.ic3.gov/CSA/2021",
    2022: "https://www.ic3.gov/CSA/2022",
    2023: "https://www.ic3.gov/CSA/2023",
    2024: "https://www.ic3.gov/CSA/2024",
    2025: "https://www.ic3.gov/CSA/2025"
}

all_rows = []

for year, page_url in urls.items():
    print(f"\nScraping {year} CSA page...")
    pdf_entries = scrape_pdf_links(page_url)
    print(f"Found {len(pdf_entries)} PDFs for {year}")

    for entry in pdf_entries:
        pdf_path = download_pdf(entry["url"], folder=f"pdfs/{year}")
        if not pdf_path:
            continue

        text = extract_text(pdf_path)
        if not text.strip():
            print(f"No text extracted from {pdf_path}")
            continue

        keyword_counts, summary = find_keywords_and_sentences(text)
        if entry["date"] is None:
            print(f"Skipping {entry['title']}: no date found")
            continue

        row = {
            "title": entry["title"],
            "date": entry["date"].strftime("%Y-%m-%d"),
            "quarter": get_quarter(entry["date"]),
            "keyword_counts": keyword_counts,
            "summary": summary
        }
        all_rows.append(row)

# Save to CSV
df = pd.DataFrame(all_rows)
df.to_csv(CSV_FILE, index=False)
print(f"\n✅ CSV created: {CSV_FILE} ({len(df)} entries)")
