import os
import asyncio
import math
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------
# Clients
# ---------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Settings
# ---------------------------
MODEL = "text-embedding-3-small"
BATCH = 50           # embeddings per batch
FETCH_LIMIT = 500    # rows to fetch per run
MAX_TEXT_LENGTH = 6000  # truncate text to control OpenAI costs

# ---------------------------
# Fetch rows without embeddings
# ---------------------------
def fetch_missing():
    res = (
        supabase.table("pdf_summaries")
        .select("id, text")
        .is_("embedding", "null")
        .limit(FETCH_LIMIT)
        .execute()
    )
    return res.data or []

# ---------------------------
# Create embeddings
# ---------------------------
async def embed_batch(batch_texts):
    # truncate text to avoid cost issues
    inputs = [(t or "")[:MAX_TEXT_LENGTH] for t in batch_texts]
    resp = client.embeddings.create(model=MODEL, input=inputs)
    return [d.embedding for d in resp.data]

# ---------------------------
# Upsert embeddings (update only)
# ---------------------------
async def upsert_embeddings(rows):
    total_batches = math.ceil(len(rows) / BATCH)

    for i in range(total_batches):
        chunk = rows[i*BATCH:(i+1)*BATCH]
        texts = [r["text"] or "" for r in chunk]
        embeddings = await embed_batch(texts)

        # Update only the embedding column for existing rows
        for j in range(len(chunk)):
            supabase.table("pdf_summaries").update({"embedding": embeddings[j]}).eq("id", chunk[j]["id"]).execute()

        print(f"✅ Batch {i+1}/{total_batches} upserted ({len(chunk)} embeddings)")

# ---------------------------
# Main execution
# ---------------------------
async def main():
    rows = fetch_missing()
    print(f"Rows needing embeddings: {len(rows)}")
    if not rows:
        print("Nothing to embed. ✅")
        return

    await upsert_embeddings(rows)
    print("✅ All embeddings processed.")

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    asyncio.run(main())
