from supabase import create_client
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Read values from environment
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url, key)

print("SUPABASE_URL:", url)
print("SUPABASE_KEY is set:", key is not None)
print("Connected:", supabase is not None)
