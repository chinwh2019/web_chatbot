import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import openai
from dotenv import load_dotenv
from openai import OpenAI


class EmbeddingDatabase:
    def __init__(self, model: str = "text-embedding-3-small"):
        openai.api_key = load_dotenv()  # Load variables from .env
        self.client = OpenAI()
        self.model = model
        self.embeddings: List[np.ndarray] = []
        self.texts: List[str] = []
        self.urls: List[str] = []

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text using OpenAI's API."""
        response = self.client.embeddings.create(input=text, model=self.model)
        return np.array(response.data[0].embedding)

    def build_from_json(self, json_path: str) -> None:
        """Build embedding database from JSON file containing text-URL pairs."""
        with open(json_path, "r") as f:
            data = json.load(f)

        for entry in data:
            # Process title and content
            if "title" in entry:
                self.texts.append(entry["title"])
                self.urls.append("")  # or some default URL
                self.embeddings.append(self.generate_embedding(entry["title"]))

            if "content" in entry:
                self.texts.append(entry["content"])
                self.urls.append("")  # or some default URL
                self.embeddings.append(self.generate_embedding(entry["content"]))

            # Process links
            if "links" in entry:
                for link in entry["links"]:
                    text = link["text"]
                    url = link["url"]

                    embedding = self.generate_embedding(text)

                    self.embeddings.append(embedding)
                    self.texts.append(text)
                    self.urls.append(url)

    def save_database(self, path: str) -> None:
        """Save the embedding database to disk."""
        data = {"embeddings": self.embeddings, "texts": self.texts, "urls": self.urls}
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_database(self, path: str) -> None:
        """Load the embedding database from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.embeddings = data["embeddings"]
        self.texts = data["texts"]
        self.urls = data["urls"]

    def find_most_similar(self, query: str, top_k: int = 1) -> List[Dict[str, str]]:
        """
        Find the most similar entries for a given query.
        Returns a list of dictionaries containing text and URL.
        """
        query_embedding = self.generate_embedding(query)

        # Calculate cosine similarity between query and all embeddings
        similarities = [
            np.dot(query_embedding, emb)
            / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            for emb in self.embeddings
        ]

        # Get indices of top_k most similar entries
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "text": self.texts[idx],
                    "url": self.urls[idx],
                    "similarity": similarities[idx],
                }
            )

        return results


def main():
    # Initialize and build the database
    db = EmbeddingDatabase()
    db.build_from_json("processed_data/all_processed_data.json")

    # # Save the database to avoid regenerating embeddings
    db.save_database("embeddings/embeddings.pkl")

    # Later, load the database
    db = EmbeddingDatabase()
    db.load_database("embeddings/embeddings.pkl")

    # Find similar entries
    results = db.find_most_similar("what is rakuten mobile", top_k=3)
    for result in results:
        print(f"Text: {result['text']}")
        print(f"URL: {result['url']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print("---")


if __name__ == "__main__":
    main()
