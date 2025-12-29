import chromadb
from sentence_transformers import SentenceTransformer
import requests

# http://musicbrainz.org/ws/2/release-group/c9fdb94c-4975-4ed6-a96f-ef6d80bb7738?inc=artist-credits+releases
fetch_url = "http://musicbrainz.org/ws/2/release-group/"

# 1. Load resources
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
client = chromadb.PersistentClient(path="./review_search_db")
collection = client.get_collection(name="music_reviews")


def query():
    while True:
        query = input("\nEnter your search (or 'q' to quit): ")
        if query.lower() == "q":
            break

        # Embed the query
        query_vec = model.encode(query).tolist()

        # Search
        results = collection.query(query_embeddings=[query_vec], n_results=5)

        # Print results
        print("\n--- TOP MATCHES ---")
        for i in range(len(results["ids"][0])):
            dist = results["distances"][0][i]
            # In Cosine space, lower distance (near 0) is better
            print(f"Score: {1 - dist:.4f}")  # Showing confidence %
            # print(f"Entity ID: {results['metadatas'][0][i]['entity_id']}")
            # print(f"Excerpt: {results['documents'][0][i][:200]}...")
            # print("-" * 20)
            
            album_obj = get_album(results['metadatas'][0][i]['entity_id'])
            print_album_details(album_obj, f"{1 - dist:.4f}")

def print_album_details(album_obj, entity_dist=None):
    # Colors for Kitty (ANSI Escape Codes)
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    title = album_obj.get('title', 'Unknown Title')
    release_date = album_obj.get('first-release-date', 'N/A')
    
    # Extract Artist Credits
    artist_credits = album_obj.get('artist-credit', [])
    artists = ", ".join([ac['name'] for ac in artist_credits])
    
    # Extract Tags (take top 5)
    tags = album_obj.get('tags', [])
    tag_list = ", ".join([t['name'] for t in tags[:5]])

    # Extract Ratings
    rating_data = album_obj.get('rating', {})
    rating_val = rating_data.get('value', 'N/A')
    votes = rating_data.get('votes-count', 0)

    print(f"\n{BOLD}{CYAN}💿 ALBUM INFORMATION{RESET}")
    print("-" * 30)
    print(f"Similarity score: {entity_dist}")
    print(f"{BOLD}Title:{RESET}    {title}")
    print(f"{BOLD}Artist:{RESET}   {GREEN}{artists}{RESET}")
    print(f"{BOLD}Released:{RESET} {release_date}")
    print(f"{BOLD}Type:{RESET}     {album_obj.get('primary-type', 'Unknown')}")
    
    if tag_list:
        print(f"{BOLD}Tags:{RESET}     {YELLOW}{tag_list}{RESET}")
    
    print(f"{BOLD}Rating:{RESET}   {rating_val}/5 ({votes} votes)")
    print("-" * 30)


def get_album(entity_id, entity_type="release-group"):
    """
    Fetches metadata from MusicBrainz for a specific MBID.
    
    Args:
        entity_id (str): The MusicBrainz ID (MBID).
        entity_type (str): The type of entity (e.g., 'release', 'release-group', 'artist').
        
    Returns:
        dict: The JSON response from MusicBrainz.
    """
    # MusicBrainz requires a descriptive User-Agent
    # Format: <AppName>/<Version> ( <ContactInfo> )
    headers = {
        "User-Agent": "MyVectorSearchApp/0.1 ( whayward@gmail.com )"
    }
    
    url = f"https://musicbrainz.org/ws/2/{entity_type}/{entity_id}"
    params = {
        "fmt": "json",
        "inc": "artists+ratings+tags" # Customize based on what you need
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        
        # MusicBrainz rate limits are strict (1 request per second)
        if response.status_code == 503:
            print("Rate limited! Sleeping...")
            time.sleep(2)
            return get_album(entity_id, entity_type)
            
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching MBID {entity_id}: {e}")
        return None


if __name__ == "__main__":
    query()
