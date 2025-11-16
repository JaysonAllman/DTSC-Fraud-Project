from supabase import create_client
from dotenv import load_dotenv
import os

# Load .env values
load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url, key)

def upload_keywords(data):
    """
    data should be a list of dictionaries like:
    [
        {"keyword": "phishing", "count": 15, "year": 2023},
        ...
    ]
    """
    response = supabase.table("fraud_keywords").insert(data).execute()
    return response

if __name__ == "__main__":
    # 🔹 Add as many rows as you want here
    sample_data = [
        {"keyword": "phishing",          "count": 120, "year": 2022},
        {"keyword": "extortion",         "count": 88,  "year": 2022},
        {"keyword": "identity theft",    "count": 132, "year": 2023},
        {"keyword": "phishing",          "count": 170, "year": 2023},
        {"keyword": "credit card fraud", "count": 98,  "year": 2024},
        {"keyword": "phishing",          "count": 120, "year": 2022},
        {"keyword": "extortion",         "count": 88,  "year": 2022},
        {"keyword": "identity theft",    "count": 132, "year": 2023},
        {"keyword": "phishing",          "count": 170, "year": 2023},
        {"keyword": "credit card fraud", "count": 98,  "year": 2024},
    ]

    result = upload_keywords(sample_data)
    print("Inserted rows:", result)
