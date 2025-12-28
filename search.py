import chromadb
from sentence_transformers import SentenceTransformer

# 1. Load resources
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
client = chromadb.PersistentClient(path="./review_search_db")
collection = client.get_collection(name="music_reviews")


def query_system():
    while True:
        query = input("\nEnter your search (or 'q' to quit): ")
        if query.lower() == "q":
            break

        # Embed the query
        query_vec = model.encode(query).tolist()

        # Search
        results = collection.query(query_embeddings=[query_vec], n_results=3)

        # Print results
        print("\n--- TOP MATCHES ---")
        for i in range(len(results["ids"][0])):
            dist = results["distances"][0][i]
            # In Cosine space, lower distance (near 0) is better
            print(f"Score: {1 - dist:.4f}")  # Showing confidence %
            print(f"Entity ID: {results['metadatas'][0][i]['entity_id']}")
            print(f"Excerpt: {results['documents'][0][i][:200]}...")
            print("-" * 20)


if __name__ == "__main__":
    query_system()
