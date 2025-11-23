DTSC Fraud Project – README

Overview

This project extracts fraud-related keywords from IC3 reports, stores them in a Supabase database, and visualizes trends through a Streamlit dashboard. The system provides insights into the most common fraud types across multiple years.

Features

• Connects securely to Supabase using .env file.
• Processes fraud keywords from uploaded PDF summaries.
• Visualizes trends with bar charts, line graphs, and heatmaps.
• Provides keyword definitions and real‑world impacts.
• Includes CSV download functionality.
• Contains a full AI‑generated summary of trends.

Files in Repository

• app_local.py – Local version with .env support.
• app.py – Cloud deployment version for Streamlit.
• requirements.txt – Dependencies for running the project.
• test_supabase_select.py – Supabase connection tester.

How to Run Locally

1. Install Python 3.10 or above.
2. Install dependencies: pip install -r requirements.txt
3. Create a .env file in the project folder with SUPABASE_URL and SUPABASE_SERVICE_KEY.
4. Run the dashboard: streamlit run app_local.py

Deployment

The dashboard deploys automatically on Streamlit Cloud whenever changes are pushed to the GitHub repository. Ensure the deployment file is named app.py.
Authors

This project was created by:

- Taylor Foster
- Sam McClure
- Jayson Allman
- Yousef Eddin



Streamlit Deployment

The project is deployed on Streamlit Cloud and updates automatically whenever changes are pushed to the GitHub repository. The live dashboard can be accessed at:

https://dtsc-fraud-project-team2.streamlit.app

To deploy manually, ensure your deployment file is named 'app.py' and located in the root of the repository. Streamlit Cloud detects this file and runs the app upon deployment.
