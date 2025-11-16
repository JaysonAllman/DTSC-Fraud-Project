DTSC Fraud Project – README
1. Project Overview
This project analyzes fraud keyword data extracted from IC3 PDF summaries. The Streamlit dashboard visualizes top fraud keywords, trends over time, and allows CSV downloads. Data is stored in a Supabase PostgreSQL database.

2. Authors
• Team Member 1: Taylor Foster
• Team Member 2: Jayson Allman
• Team Member 3: Yousef Eddin
• Team Member 4: Sam McClure

3. Project Structure
• app.py – Streamlit dashboard
• loader.py – Inserts keyword data into Supabase
• scraper.py – Extracts text from PDFs (optional)
• crime_keywords.py – Keyword logic
• pdf_summaries.csv – Extracted summary dataset
• requirements.txt – Python dependencies
• .env – Local Supabase credentials (not included in repo)

4. Installation
1. Clone the repository:
   git clone https://github.com/JaysonAllman/DTSC-Fraud-Project.git

2. Navigate to the project folder:
   cd DTSC-Fraud-Project

3. Install dependencies:
   pip install -r requirements.txt

5. Environment Variables
Create a .env file in the root directory:

SUPABASE_URL="your-url-here"
SUPABASE_KEY="your-service-role-key-here"

These values must also be added to Streamlit Cloud under 'Secrets'.
6. Running Locally
Run the Streamlit app:
    streamlit run app.py

7. Deployment Instructions (Streamlit Cloud)
1. Push code to GitHub.
2. Go to https://share.streamlit.io
3. Select 'New app' and choose this repository.
4. Set app file to app.py.
5. Add secrets in Settings → Secrets:
   SUPABASE_URL="..."
   SUPABASE_KEY="..."
6. Deploy the application.
8. Data Flow
• scraper.py extracts summaries → pdf_summaries.csv
• loader.py sends structured keyword data to Supabase
• app.py pulls data from Supabase → displays charts
