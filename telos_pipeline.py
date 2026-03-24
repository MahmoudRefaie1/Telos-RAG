"""
telos_pipeline.py
Full end-to-end: User query → RAG retrieves memories → Llama responds
"""

import requests
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ── RAG Setup ──────────────────────────────────────────────
CHROMA_PATH = "./chroma_local"
LLM_URL     = "http://10.7.57.198:8000/chat"   # ← replace with their IP

local_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client   = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name="telos_memories_local",
    embedding_function=local_ef,
    metadata={"hnsw:space": "cosine"},
)

# ── Step 1: Retrieve memories from ChromaDB ─────────────────
def get_memories(user_id: str, query: str, eeg_state: str = None, n: int = 3) -> list:
    where_filter = {"user_id": user_id}
    if eeg_state:
        where_filter = {
            "$and": [
                {"user_id":  {"$eq": user_id}},
                {"eeg":      {"$eq": eeg_state}},
            ]
        }

    results = collection.query(
        query_texts=[query],
        n_results=n,
        where=where_filter,
        include=["documents", "metadatas"],
    )
    return results["documents"][0]

# ── Step 2: Build system prompt ─────────────────────────────
def build_system_prompt(memories: list) -> str:
    memory_block = "\n".join(f"  {i+1}. {m}" for i, m in enumerate(memories))
    return f"""You are Telos, a supportive mental health companion for students.
You help students manage stress, burnout, and academic pressure.
Always be empathetic and personalized.

Relevant memories about this user:
{memory_block}

Use these memories to give a personalized response."""

# ── Step 3: Send to Llama API ───────────────────────────────
def ask_llama(system_prompt: str, user_query: str) -> str:
    payload = {
        "system": system_prompt,
        "user":   user_query,
    }
    response = requests.post(LLM_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["response"]

# ── Full Pipeline ───────────────────────────────────────────
def ask_telos(user_id: str, query: str, eeg_state: str = None) -> str:
    print(f"\n[1] Query    : {query}")
    print(f"[1] EEG state: {eeg_state}")

    memories = get_memories(user_id, query, eeg_state)
    print(f"[2] Retrieved memories:")
    for m in memories:
        print(f"    - {m}")

    system_prompt = build_system_prompt(memories)
    print(f"[3] Sending to Llama at {LLM_URL}...")

    answer = ask_llama(system_prompt, query)
    print(f"[4] Telos says: {answer}")
    return answer


# ── Test ────────────────────────────────────────────────────
if __name__ == "__main__":
    ask_telos(
        user_id   = "mahmoud",
        query     = "I feel burnt out from studying",
        eeg_state = "high_stress",
    )
