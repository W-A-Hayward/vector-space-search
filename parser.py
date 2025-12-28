import json
import re
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Init Model & DB
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
client = chromadb.PersistentClient(path="./review_search_db")

# Delete old collection if it exists to ensure Cosine setting is applied
try:
    client.delete_collection("music_reviews")
except:
    pass

collection = client.get_or_create_collection(
    name="music_reviews", metadata={"hnsw:space": "cosine"}
)


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[\*\#\_]", "", text)
    text = re.sub(r"[\n\r\t]+", " ", text)
    return text.strip()


def stream_reviews(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


# 2. Batch Processing
batch_size = 128
current_batch = []
global_count = 0

print("Starting embedding process...")
for review in stream_reviews("reviews.jsonl"):
    cleaned = clean_text(review.get("text", ""))
    if not cleaned:
        continue

    review["cleaned_text"] = cleaned
    current_batch.append(review)

    if len(current_batch) >= batch_size:
        texts = [r["cleaned_text"] for r in current_batch]
        vectors = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            embeddings=vectors,
            ids=[
                f"{r['entity_id']}_{global_count + i}"
                for i, r in enumerate(current_batch)
            ],
            metadatas=[{"entity_id": r["entity_id"]} for r in current_batch],
            documents=texts,
        )
        global_count += len(current_batch)
        print(f"Indexed {global_count} reviews...")
        current_batch = []

# Final partial batch
if current_batch:
    texts = [r["cleaned_text"] for r in current_batch]
    vectors = model.encode(texts).tolist()
    collection.add(
        embeddings=vectors,
        ids=[
            f"{r['entity_id']}_{global_count + i}" for i, r in enumerate(current_batch)
        ],
        metadatas=[{"entity_id": r["entity_id"]} for r in current_batch],
        documents=texts,
    )

print("Done indexing.")
