#This is just a simple program to test how to use chromadb for storing and querying memories.
#So I added 9 memories and then queried for the 3 most similar memories to "I feel burnt out from studying". 
#The results are printed with their IDs, text, and distance from the query.

import chromadb

# Creating and testing  chromadb.
client = chromadb.Client()
collection = client.create_collection("test_memories")

# Storing some memories
collection.add(
    documents=["I felt anxious before midterms", 
               "Painting helped me relax",
               "I want to learn Spanish",
               "I enjoy hiking in the mountains",
               "Cooking is a great stress reliever",
               "I have a fear of public speaking",
               "I find comfort in listening to music",
               "Last Friday, I went to the gym which made me feel energized for my academic work",
               "I struggle with time management"],
    ids=["mem1", "mem2", "mem3", "mem4", "mem5", "mem6", "mem7", "mem8", "mem9"]
)

results = collection.query(
    query_texts=["I feel burnt out from studying"],
    n_results=3
)


for doc_id, doc_text, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["distances"][0]
    ):
    print(f"- ID: {doc_id}")
    print(f"  Text: {doc_text}")
    print(f"  Distance: {dist:.3f}\n")
