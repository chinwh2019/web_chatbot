import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from settings import settings
from utils.exceptions import EmbeddingError
from utils.logger import setup_logger

logger = setup_logger("embeddings", "embeddings.log")


class EmbeddingDatabase:
    def __init__(self):
        """Initialize the EmbeddingDatabase using settings configuration."""
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.EMBEDDING_MODEL
        self._reset_database()

    def _reset_database(self) -> None:
        """Reset the database to empty state."""
        self.embeddings: List[np.ndarray] = []
        self.texts: List[str] = []
        self.urls: List[str] = []

    def _validate_entry(self, entry: Dict) -> bool:
        """Validate if the entry contains required fields."""
        return any(key in entry for key in ["title", "content", "links"])

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text using OpenAI's API."""
        try:
            response = self.client.embeddings.create(input=text, model=self.model)
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise EmbeddingError(f"Embedding generation failed: {str(e)}")

    def build_from_json(self, json_path: Optional[Path] = None) -> None:
        """Build embedding database from JSON file."""
        try:
            if json_path is None:
                json_path = settings.PROCESSED_DATA_DIR / "combined.json"

            if not json_path.exists():
                raise FileNotFoundError(f"JSON file not found: {json_path}")

            # Reset existing database
            self._reset_database()

            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            logger.info(f"Building database from {json_path}")
            for entry in data:
                if self._validate_entry(entry):
                    self._process_entry(entry)
                else:
                    logger.warning(f"Skipping invalid entry: {entry}")

        except Exception as e:
            logger.error(f"Failed to build database: {str(e)}")
            raise EmbeddingError(f"Database building failed: {str(e)}")

    def _process_entry(self, entry: Dict) -> None:
        """Process a single entry and add it to the database."""
        try:
            if "title" in entry:
                embedding = self.generate_embedding(entry["title"])
                self.embeddings.append(embedding)
                self.texts.append(entry["title"])
                self.urls.append("")

            if "content" in entry:
                embedding = self.generate_embedding(entry["content"])
                self.embeddings.append(embedding)
                self.texts.append(entry["content"])
                self.urls.append("")

            if "links" in entry:
                for link in entry["links"]:
                    embedding = self.generate_embedding(link["text"])
                    self.embeddings.append(embedding)
                    self.texts.append(link["text"])
                    self.urls.append(link["url"])

        except Exception as e:
            logger.error(f"Error processing entry: {str(e)}")
            raise EmbeddingError(f"Entry processing failed: {str(e)}")

    def save_database(self, path: Optional[Path] = None) -> None:
        """Save the embedding database to disk."""
        try:
            if path is None:
                path = settings.EMBEDDINGS_DIR / "embeddings.pkl"

            path.parent.mkdir(parents=True, exist_ok=True)

            with path.open("wb") as f:
                pickle.dump(
                    {
                        "embeddings": self.embeddings,
                        "texts": self.texts,
                        "urls": self.urls,
                        "model": self.model,
                    },
                    f,
                )
            logger.info(f"Database saved successfully to {path}")

        except Exception as e:
            logger.error(f"Failed to save database: {str(e)}")
            raise EmbeddingError(f"Database saving failed: {str(e)}")

    def load_database(self, path: Optional[Path] = None) -> None:
        """Load the embedding database from disk."""
        try:
            if path is None:
                path = settings.EMBEDDINGS_DIR / "embeddings.pkl"

            with path.open("rb") as f:
                data = pickle.load(f)

            self.embeddings = data["embeddings"]
            self.texts = data["texts"]
            self.urls = data["urls"]
            self.model = data.get("model", self.model)

            logger.info(f"Database loaded successfully from {path}")

        except Exception as e:
            logger.error(f"Failed to load database: {str(e)}")
            raise EmbeddingError(f"Database loading failed: {str(e)}")

    def find_most_similar(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Find the most similar entries for a given query."""
        try:
            if not self.embeddings:
                raise EmbeddingError("Database is empty")

            if top_k is None:
                top_k = settings.TOP_K_RESULTS

            query_embedding = self.generate_embedding(query)

            # Vectorized similarity calculation
            query_norm = np.linalg.norm(query_embedding)
            embeddings_norm = np.linalg.norm(self.embeddings, axis=1)
            similarities = np.dot(self.embeddings, query_embedding) / (
                embeddings_norm * query_norm
            )

            top_indices = np.argsort(similarities)[-top_k:][::-1]

            results = [
                {
                    "text": self.texts[idx],
                    "url": self.urls[idx],
                    "similarity": float(similarities[idx]),
                }
                for idx in top_indices
            ]

            logger.debug(f"Found {len(results)} results for query: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise EmbeddingError(f"Search operation failed: {str(e)}")


def main():
    try:
        # Initialize database
        db = EmbeddingDatabase()

        # Build database (this will reset existing embeddings)
        db.build_from_json()

        # Save database
        db.save_database()

        # Load database
        db = EmbeddingDatabase()
        db.load_database()

        # Search
        results = db.find_most_similar("what is rakuten mobile")

        for result in results:
            print(f"Text: {result['text']}")
            print(f"URL: {result['url']}")
            print(f"Similarity: {result['similarity']:.4f}")
            print("---")

    except EmbeddingError as e:
        logger.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
