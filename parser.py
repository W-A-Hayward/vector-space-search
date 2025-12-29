import json
import re
import chromadb
from sentence_transformers import SentenceTransformer
import torch

# --- 1. SETUP & DEFINITIONS (Safe to keep global) ---
def clean_text(text):
    if not text: return ""
    text = re.sub(r"https?://\S+|[\*\#\_]|[\n\r\t]+", " ", text)
    return text.strip()

def stream_reviews(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

# --- 2. MAIN EXECUTION BLOCK (This prevents the crash) ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device=device
    )

    client = chromadb.PersistentClient(path="./review_search_db")
    collection = client.get_or_create_collection(
        name="music_reviews", 
        metadata={"hnsw:space": "cosine"}
    )

    DB_BATCH_SIZE = 5000 
    MODEL_BATCH_SIZE = 128 
    
    current_batch_texts = []
    current_batch_metas = []
    current_batch_ids = []
    global_count = 0

    print(f"Starting embedding process on {device.upper()}...")

    # Multi-process pool must be inside the __main__ check
    pool = None
    if device == "cpu":
        pool = model.start_multi_process_pool()

    try:
        for review in stream_reviews("reviews.jsonl"):
            cleaned = clean_text(review.get("text", ""))
            if not cleaned or len(cleaned) < 10:
                continue

            current_batch_texts.append(cleaned)
            current_batch_ids.append(f"{review.get('entity_id')}_{global_count}")
            current_batch_metas.append({"entity_id": review.get("entity_id")})
            global_count += 1

            if len(current_batch_texts) >= DB_BATCH_SIZE:
                # The new way: just use model.encode
                # If pool is None (e.g., on GPU), it handles it automatically
                vectors = model.encode(
                    current_batch_texts, 
                    pool=pool,               # Pass the pool here
                    batch_size=MODEL_BATCH_SIZE,
                    show_progress_bar=False
                ).tolist()

                collection.add(
                    embeddings=vectors,
                    ids=current_batch_ids,
                    metadatas=current_batch_metas,
                    documents=current_batch_texts,
                )
                
                print(f"\rIndexed {global_count} reviews...", end="", flush=True)
                current_batch_texts, current_batch_metas, current_batch_ids = [], [], []        # Handle final partial batch
                if current_batch_texts:
                    vectors = model.encode(current_batch_texts).tolist()
                    collection.add(embeddings=vectors, ids=current_batch_ids, metadatas=current_batch_metas, documents=current_batch_texts)

    finally:
        # Crucial: always stop the pool even if the script crashes
        if pool:
            model.stop_multi_process_pool(pool)

    print("\nDone indexing.")
