"""
Telos RAG memory (Chroma) — single-file example

Didnt work cus our quota for text-embedding-3-small was used up. You can switch to text-embedding-3-large or use your own OpenAI key with quota.
"""

import os
from datetime import datetime, timezone

import chromadb 
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


# -----------------------------
# 1) Configure persistence + embeddings
# -----------------------------
CHROMA_PATH = "./chroma"  # folder where Chroma persists data
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # set in your env

if not OPENAI_API_KEY:
    raise RuntimeError(
        "Missing OPENAI_API_KEY. Set it in your environment, e.g.\n"
        "export OPENAI_API_KEY='...'\n"
    )

openai_ef = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small",  # or "text-embedding-3-large"
)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name="telos_memories",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},  # cosine distance is common for embeddings
)


# -----------------------------
# 2) Seed data (optional)
# -----------------------------
memories = [
    {"id": "m1", "text": "Felt anxious before midterms, painting helped me relax", "eeg": "high_stress"},
    {"id": "m2", "text": "High alpha detected during Spanish practice, needed break", "eeg": "low_focus"},
    {"id": "m3", "text": "Walking outside helped after burnout from studying", "eeg": "high_stress"},
    {"id": "m4", "text": "Goal: learn Spanish 30 minutes daily", "eeg": "focused"},
]

USER_ID = "mahmoud"

# Prevent duplicates if you run the script multiple times
existing = collection.get(include=[])
existing_ids = set(existing.get("ids", []))

to_add = [m for m in memories if m["id"] not in existing_ids]

if to_add:
    now_iso = datetime.now(timezone.utc).isoformat()

    collection.add(
        documents=[m["text"] for m in to_add],
        ids=[m["id"] for m in to_add],
        metadatas=[
            {
                "eeg": m["eeg"],
                "user_id": USER_ID,
                "created_at": now_iso,  # useful for recency/decay later
            }
            for m in to_add
        ],
    )
    print(f"Added {len(to_add)} new memories to Chroma at '{CHROMA_PATH}'.")
else:
    print("No new memories to add (IDs already exist).")


# -----------------------------
# 3) Query (RAG retrieval)
# -----------------------------
query = "I feel burnt out from studying"

results = collection.query(
    query_texts=[query],
    n_results=2,
    where={"user_id": USER_ID},
    include=["documents", "metadatas", "distances"],  # be explicit/reliable
)

print(f"\nQuery: '{query}'\n")

ids = results["ids"][0]
docs = results["documents"][0]
dists = results["distances"][0]
metas = results["metadatas"][0]

for i, (doc_id, doc_text, dist, meta) in enumerate(zip(ids, docs, dists, metas), start=1):
    # NOTE: 'dist' is a DISTANCE, so smaller = more similar.
    # For cosine distance, an approximate similarity can be shown as (1 - dist).
    sim = 1 - dist

    print(f"Top {i} (id={doc_id}): {doc_text}")
    print(f"  EEG state: {meta.get('eeg')}")
    print(f"  Cosine distance: {dist:.3f}  (≈ similarity {sim:.3f})")
    print(f"  created_at: {meta.get('created_at')}\n")
    
    