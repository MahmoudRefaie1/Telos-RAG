"""
rag_connector.py
Bridge between Telos RAG (ChromaDB) and the LLM team's Llama model.

LLM team: call get_rag_context() to get memories,
          then pass the returned prompt_block into your Llama system prompt.
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# -----------------------------
# Setup (same config as TelosRAG_LOCAL)
# -----------------------------
CHROMA_PATH = "./chroma_local"

local_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name="telos_memories_local",
    embedding_function=local_ef,
    metadata={"hnsw:space": "cosine"},
)


# -----------------------------
# Core RAG function (LLM team calls this)
# -----------------------------
def get_rag_context(user_id: str, query: str, eeg_state: str = None, n: int = 3) -> dict:
    """
    Retrieves relevant memories for a user query.

    Args:
        user_id   : the user's ID (e.g. "mahmoud")
        query     : what the user just said (e.g. "I feel burnt out")
        eeg_state : optional EEG filter (e.g. "high_stress", "low_focus", "focused")
        n         : how many memories to retrieve (default 3)

    Returns a dict with:
        - memories      : list of relevant memory strings
        - prompt_block  : ready-to-paste string for Llama system prompt
        - metadata      : list of metadata dicts (timestamps, eeg states)
    """

    # Build filter
    where_filter = {"user_id": user_id}
    if eeg_state:
        where_filter = {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"eeg": {"$eq": eeg_state}},
            ]
        }

    results = collection.query(
        query_texts=[query],
        n_results=n,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    memories  = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Format a ready-to-use block for Llama system prompt
    prompt_block = "Relevant memories about this user:\n"
    for i, (mem, meta) in enumerate(zip(memories, metadatas), start=1):
        prompt_block += f"  {i}. [{meta.get('eeg', 'unknown')} state] {mem}\n"

    return {
        "memories":     memories,
        "prompt_block": prompt_block,
        "metadata":     metadatas,
    }


# -----------------------------
# Build full Llama prompt (optional helper for LLM team)
# -----------------------------
def build_llama_prompt(user_id: str, query: str, eeg_state: str = None) -> dict:
    """
    Returns a fully formatted prompt ready for Llama.

    Returns:
        {
            "system": "...",   # pass as system message
            "user":   "...",   # pass as user message
        }
    """
    context = get_rag_context(user_id, query, eeg_state)

    system_prompt = f"""You are Telos, a supportive mental health companion for students.
You help students manage stress, burnout, and academic pressure.
Always be empathetic, concise, and personalized.

{context['prompt_block']}
Use these memories to give a personalized response. 
If no memories are relevant, respond generally but warmly."""

    return {
        "system": system_prompt,
        "user":   query,
    }


# -----------------------------
# Test it (run this file directly to verify)
# -----------------------------
if __name__ == "__main__":
    TEST_USER  = "mahmoud"
    TEST_QUERY = "I feel burnt out from studying"
    TEST_EEG   = "high_stress"

    print("=" * 60)
    print("TEST: get_rag_context()")
    print("=" * 60)
    ctx = get_rag_context(TEST_USER, TEST_QUERY, TEST_EEG)
    print("Memories retrieved:")
    for m in ctx["memories"]:
        print(f"  - {m}")

    print("\n" + "=" * 60)
    print("TEST: build_llama_prompt()")
    print("=" * 60)
    prompt = build_llama_prompt(TEST_USER, TEST_QUERY, TEST_EEG)
    print("--- SYSTEM PROMPT ---")
    print(prompt["system"])
    print("--- USER MESSAGE ---")
    print(prompt["user"])
