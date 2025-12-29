import requests
import json
import time

def fetch_all_critiquebrainz_reviews():
    url = "https://critiquebrainz.org/ws/1/review/"
    file_path = "reviews.jsonl"
    
    limit = 50
    offset = 0
    total_count = 1  # Placeholder to start the loop
    page = 1

    with open(file_path, "w", encoding="utf-8") as f:
        # Continue as long as our offset hasn't reached the total available reviews
        while offset < total_count:
            params = {"limit": limit, "offset": offset, "entity_type": "release_group", "review_type": "review", "include_metada": True}
            print(f"\rFetching page {page} (Offset: {offset})...")

            try:
                response = requests.get(url, params=params)
                response.raise_for_status() # Raises an error for bad status codes
                
                data = response.json()
                
                # Update the total count from the API's actual data
                total_count = data.get("count", 0)
                reviews = data.get("reviews", [])

                if not reviews:
                    break

                for review in reviews:
                    out_data = {
                        "id": review.get("id"),
                        "entity_id": review.get("entity_id"),
                        "text": review.get("text"),
                        "rating": review.get("rating"),
                    }
                    f.write(json.dumps(out_data) + "\n")

                # Prepare for the next page
                offset += limit
                page += 1
                
                # Rate limiting
                time.sleep(2)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching data: {e}")
                break

    print(f"\nFinished! Total reviews found: {total_count}")

if __name__ == "__main__":
    fetch_all_critiquebrainz_reviews()
