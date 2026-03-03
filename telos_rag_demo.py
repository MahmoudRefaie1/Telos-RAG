import chromadb
from langchain_openai import OpenAIEmbeddings 

# Telos memories
memories = [
    {"id": "m1", "text": "Felt anxious before midterms, painting helped me relax", "eeg": "high_stress"},
    {"id": "m2", "text": "High alpha detected during Spanish practice, needed break", "eeg": "low_focus"},
    {"id": "m3", "text": "Walking outside helped after burnout from studying", "eeg": "high_stress"},
    {"id": "m4", "text": "Goal: learn Spanish 30 minutes daily", "eeg": "focused"},
]

client = chromadb.Client()
collection = client.get_or_create_collection("telos_memories")

# Add memories with metadata
collection.add(
    documents=[m["text"] for m in memories],
    ids=[m["id"] for m in memories],
    metadatas=[{"eeg": m["eeg"], "user_id": "mahmoud"} for m in memories]
)

# Test query
query = "I feel burnt out from studying"
results = collection.query(
    query_texts=[query],
    n_results=2,
    where={"user_id": "mahmoud"}  # only Mahmoud's memories
)

print(f"Query: '{query}'\n")
for i, (doc_id, doc_text, dist) in enumerate(zip(results["ids"][0], results["documents"][0], results["distances"][0])):
    print(f"Top {i+1}: {doc_text}")
    print(f"  EEG state: {results['metadatas'][0][i]['eeg']}")
    print(f"  Similarity: {dist:.3f}\n")