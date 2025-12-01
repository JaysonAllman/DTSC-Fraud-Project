# Fraud Trends Dashboard
**Tagline:** Interactive AI-powered insights into fraud patterns from IC3 reports

## Authors
- Samuel McClure, Taylor Foster, Jayson Allman and Yousef Eddin  

## Project Summary
This project automates the collection, analysis, and summarization of fraud reports from IC3, enabling anyone to track trends, detect emerging threats, and generate actionable insights efficiently.

## Quick Start

Follow these steps to get the Fraud Analysis Dashboard running locally.

---

### 1. Clone the Repository
```bash
git clone https://github.com/JaysonAllman/DTSC-Fraud-Project.git
cd DTSC-Fraud-Project
```
### 2. Create a Virtual Environment and Install Dependencies
```bash
uv venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
uv pip install -r requirements.txt
```

### 3. Setup Environment Variables
- Create a .env file and insert these:
- SUPABASE_URL=<your_supabase_url_here>
- SUPABASE_KEY=<your_supabase_anon_key_here>
- OPENAI_API_KEY=<your_openai_api_key_here>

### 4. Run the Streamlit Dashboard
```bash
uv run streamlit run app.py
```

## Architecture Diagram

The workflow of the **DTSC Fraud Analysis Project** is illustrated below. It shows the ETL pipeline from scraping IC3 webpages and PDFs, generating keyword summaries, storing data in Supabase, and visualizing trends on the Streamlit dashboard.

```mermaid
graph TD
    A[IC3 Website] -->|Scrape PDFs & HTML| B[Scraper.py]
    B --> C[fraud_summaries.csv]
    C --> D[LLM Reports / fraud_reports.csv]
    C --> E[Supabase: pdf_summaries table]
    D --> E[Supabase: fraud_reports table]
    E --> F[Streamlit Dashboard]
```
## Dashboard Example
<img width="700" alt="Dashboard Screenshot 1" src="https://github.com/JaysonAllman/DTSC-Fraud-Project/blob/README-Test/images/Screenshot%202025-11-30%20194102.png">
<img width="700" alt="Dashboard Screenshot 2" src="https://github.com/JaysonAllman/DTSC-Fraud-Project/blob/README-Test/images/Screenshot%202025-11-30%20194353.png">

DTSC-Fraud-Project/
├─ app.py                  # Streamlit dashboard
├─ scraper.py              # Scrapes IC3 PDFs & webpages
├─keywords.py              # Fraud keyword dictionary & regex
├─llm_reports.py           # Generates LLM summaries
├─loader.py                # Uploads CSV data to Supabase
├─requirements.txt         # Dependencies
├─.env                     # Environment variables (not tracked)
├─images/                  # Screenshots or GIFs
└─data/                    # Raw or processed CSV files
