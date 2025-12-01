# Fraud Trends Dashboard

## Team Member
- Jayson Allman  

**Project for Cybersecurity Data Analysis / Fraud Detection**

---

## Project Summary
This repository contains the **Fraud Trends Dashboard**, which scrapes IC3 reports, extracts fraud-related keywords, and generates AI-powered summaries for interactive visualization of fraud trends across time.

The project helps analysts quickly detect patterns, recurring fraud types, and emerging threats without manually reviewing PDFs.

---

## ⚙️ ETL Pipeline

The system follows a full **Extract–Transform–Load (ETL)** workflow:

| Stage | Description | Tools Used |
|-------|------------|------------|
| **Extract** | Downloads IC3 webpages and PDFs for fraud reports. | `requests`, `BeautifulSoup4`, `PyMuPDF` / `PyPDF2` |
| **Transform** | Cleans text, detects keywords, flattens nested JSON counts, and standardizes date formats. | `pandas`, `numpy`, `re`, `datetime` |
| **Load** | Uploads structured data to **Supabase**, storing both summaries and keyword counts. | `supabase-py`, `.env` for credentials |

### 📊 ETL Workflow
