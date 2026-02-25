import chromadb

# Creating and testing  chromadb.
client = chromadb.Client()
collection = client.create_collection("test_memories")

# Store 3 memories
collection.add(
    documents=["I felt anxious before midterms", 
               "Painting helped me relax",
               "I want to learn Spanish"],
    ids=["mem1", "mem2", "mem3"]
)

# Retrieve similar memory
results = collection.query(
    query_texts=["I feel burnt out from studying"],
    n_results=1
)

print("Retrieved:", results)
