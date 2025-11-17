import os
import re
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import pandas as pd
from datetime import datetime
from urllib.parse import urljoin

from crime_keywords import FRAUD_REGEX  # keep original import

PDF_FOLDER = "pdfs"  # keep original folder

# -------------------------------
# Helper functions
# -------------------------------

def get_quarter(date):
    month = date.month
    if month <= 3: return 1
    elif month <= 6: return 2
    elif month <= 9: return 3
    else: return 4

def is_meaningful_sentence(sentence: str) -> bool:
    s = sentence.strip()
    s = re.sub(r'\s+', ' ', s)
    if len(s.split()) < 5 or len(s) < 30 or len(s) > 1000:
        return False
    non_alpha_ratio = sum(1 for c in s if not c.isalpha() and c != ' ') / len(s)
    if non_alpha_ratio > 0.4:
        return False
    junk_patterns = [
        r"\$HTTP_PORTS", r"alert tcp", r"sid:\d+", r"http_header", r"http_uri",
        r"flowbits", r"pcre:", r"\|[0-9A-Fa-f]{2}\|",
        r"rev:\d+", r"msg:", r"metadata:", r"classtype:", r"content:"
    ]
    if any(re.search(p, s, re.IGNORECASE) for p in junk_patterns):
        return False
    if re.search(r"https?://|www\.|\.com|\.gov|\.ru|\.top|\.net", s):
        return False
    if re.search(r"[;{}<>@|$]", s):
        return False
    return True

def scrape_pdf_links(page_url):
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
            date_text = ""
            parent = a_tag.find_parent()
            if parent:
                date_text = parent.get_text(" ", strip=True)
            date_match = re.search(r"\b\w{3},\s+\d{1,2}\s+\w{3}\s+\d{4}\b", date_text)
            date_obj = datetime.strptime(date_match.group(0), "%a, %d %b %Y") if date_match else None
            pdf_entries.append({"title": title, "url": full_url, "date": date_obj})
    return pdf_entries

def download_pdf(url, folder=PDF_FOLDER):
    os.makedirs(folder, exist_ok=True)
    filename = os.path.basename(url)
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        return path
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
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + " "
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def find_keywords_and_sentences(text):
    counts = {}
    keyword_counts = {ftype: {} for ftype in FRAUD_REGEX.keys()}
    summary_sentences = []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        if not is_meaningful_sentence(sent):
            continue
        for fraud_type, pattern in FRAUD_REGEX.items():
            matches = pattern.findall(sent)
            # flatten tuples in case regex uses groups
            matches = [m if isinstance(m, str) else next(filter(None, m)) for m in matches]
            matches = [m.lower() for m in matches]

            if matches:
                counts[fraud_type] = counts.get(fraud_type, 0) + len(matches)
                for m in matches:
                    keyword_counts[fraud_type][m] = keyword_counts[fraud_type].get(m, 0) + 1
                summary_sentences.append(sent.strip())

    summary_sentences = list(dict.fromkeys(summary_sentences))
    summary_text = " ".join(summary_sentences).replace("\n", " ").replace("\r", " ")
    return counts, keyword_counts, summary_text

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
        pdf_path = download_pdf(entry["url"], folder=f"{PDF_FOLDER}/{year}")
        if not pdf_path:
            continue

        text = extract_text(pdf_path)
        if not text.strip():
            print(f"No text extracted from {pdf_path}")
            continue

        fraud_counts, keyword_counts, summary = find_keywords_and_sentences(text)
        if entry["date"] is None:
            print(f"Skipping {entry['title']}: no date found")
            continue

        row = {
            "title": entry["title"],
            "date": entry["date"].strftime("%Y-%m-%d"),
            "quarter": get_quarter(entry["date"]),
            "fraud_type_counts": fraud_counts,
            "keyword_counts": keyword_counts,
            "summary": summary
        }
        all_rows.append(row)

df = pd.DataFrame(all_rows)
df.to_csv(CSV_FILE, index=False)
print(f"\n✅ CSV created: {CSV_FILE} ({len(df)} entries)")
