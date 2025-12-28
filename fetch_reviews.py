import requests
import json
import time


def fetch_critiquebrainz_reviews(limit=50, total_pages=10):
    url = "https://critiquebrainz.org/ws/1/review/"
    file_path = "reviews.jsonl"

    with open(file_path, "w", encoding="utf-8") as f:
        for page in range(total_pages):
            params = {"limit": limit, "offset": page * limit, "format": "json"}
            print(f"Fetching page {page + 1}...")

            response = requests.get(url, params=params)
            if response.status_code != 200:
                print(f"Error fetching data: {response.status_code}")
                break

            data = response.json()
            reviews = data.get("reviews", [])

            for review in reviews:
                # Standardize the structure for our parser
                out_data = {
                    "id": review.get("id"),
                    "entity_id": review.get("entity_id"),
                    "text": review.get("text"),
                    "rating": review.get("rating"),
                }
                f.write(json.dumps(out_data) + "\n")

            # Respect the API rate limits
            time.sleep(1)


if __name__ == "__main__":
    fetch_critiquebrainz_reviews()
