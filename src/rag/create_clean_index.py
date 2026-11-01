import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("INDEX_NAME")

print(f"Checking index: {index_name}...")

# 1. Purana document-schema wala index delete karein
all_indexes = [idx["name"] for idx in pc.list_indexes()]
if index_name in all_indexes:
    print(f"Purana index '{index_name}' delete ho raha hai...")
    pc.delete_index(index_name)
    print("Delete complete.")

# 2. Pure raw vector index banayein (zero schema)
print(f"Naya clean serverless index '{index_name}' ban raha hai...")
pc.create_index(
    name=index_name,
    dimension=3072,  # gemini-embedding-001 ke liye
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

print("Badhai ho! Bina schema wala pure vector index ban gaya hai.")