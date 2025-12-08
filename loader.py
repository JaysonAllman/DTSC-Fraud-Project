import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------
# CSV Upload Function
# -------------------------------
def upload_csv(file_path, table_name, conflict_key=None):
    """
    Upload a CSV to a Supabase table using upsert.

    file_path: str → path to CSV
    table_name: str → Supabase table name
    conflict_key: str → column to match for upsert (e.g., "id", "title", "url")
    """
    # Read CSV
    df = pd.read_csv(file_path)
    print(f"Uploading {file_path} to table '{table_name}'...")

    # Fill NaN with None for Supabase compatibility
    df = df.where(pd.notnull(df), None)

    # Convert to list of dictionaries
    data = df.to_dict(orient="records")

    # Upsert or insert
    if conflict_key:
        response = supabase.table(table_name).upsert(data, on_conflict=conflict_key).execute()
    else:
        response = supabase.table(table_name).insert(data).execute()

    print(f"Uploaded {len(data)} rows to '{table_name}'")
    return response

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    # Upload fraud reports
    fraud_csv = "fraud_reports_supabase.csv"
    upload_csv(fraud_csv, "fraud_reports", conflict_key=None)

    # Upload PDF summaries
    pdf_csv = "pdf_summaries_supabase.csv"
    upload_csv(pdf_csv, "pdf_summaries", conflict_key=None)
