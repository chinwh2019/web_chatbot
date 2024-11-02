from typing import Dict, List, Optional
import logging
import json
from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from tqdm import tqdm
import numpy as np

from .utils.exceptions import EmbeddingError
from .settings import settings
from .models import Base, Document

logger = logging.getLogger(__name__)


class EmbeddingDatabase:
    def __init__(self):
        """Initialize the EmbeddingDatabase."""
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.EMBEDDING_MODEL
        
        # Initialize PostgreSQL connection
        self.engine = create_engine(settings.DATABASE_URL)
        
        Base.metadata.create_all(self.engine)  # Create tables if they don't exist
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def load_json_data(self, file_path: str) -> List[Dict]:
        """Load JSON data from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file: {str(e)}")
            raise

    def generate_embedding(self, textdata: str) -> np.ndarray:
        """Generate embedding for the given text."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=textdata
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")

    def _process_entry(self, entry: Dict) -> None:
        """Process a single entry and add it to the database."""
        try:
            documents = []
            
            if "title" in entry:
                embedding = self.generate_embedding(entry["title"])
                documents.append(Document(
                    text=entry["title"],
                    url="",
                    embedding=embedding,
                    doc_type="title"
                ))

            if "content" in entry:
                embedding = self.generate_embedding(entry["content"])
                documents.append(Document(
                    text=entry["content"],
                    url="",
                    embedding=embedding,
                    doc_type="content"
                ))

            if "links" in entry:
                for link in entry["links"]:
                    embedding = self.generate_embedding(link["text"])
                    documents.append(Document(
                        text=link["text"],
                        url=link["url"],
                        embedding=embedding,
                        doc_type="link"
                    ))

            self.session.bulk_save_objects(documents)
            self.session.commit()

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error processing entry: {str(e)}")
            raise EmbeddingError(f"Entry processing failed: {str(e)}")

    def find_most_similar(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, str]]:
        """Find the most similar entries for a given query."""
        try:
            if top_k is None:
                top_k = settings.TOP_K_RESULTS

            query_embedding = self.generate_embedding(query)
            
            # Calculate cosine similarity using dot product and norms
            results = self.session.query(Document).all()
            similarities = []
            
            for doc in results:
                doc_embedding = np.array(doc.embedding)
                # Cosine similarity = dot product / (norm1 * norm2)
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append((doc, similarity))
            
            # Sort by similarity (highest first) and take top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_results = similarities[:top_k]

            return [
                {
                    "text": doc.text,
                    "url": doc.url,
                    "similarity": float(similarity)
                }
                for doc, similarity in top_results
            ]

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise EmbeddingError(f"Search operation failed: {str(e)}")

    def __del__(self):
        """Cleanup database connection."""
        if hasattr(self, 'session'):
            self.session.close()


def main():
    try: 
        db = EmbeddingDatabase()

        logger.info("Initializing database connection...")
        db = EmbeddingDatabase()
        
        # Load JSON data
        logger.info("Loading JSON data...")
        data = db.load_json_data('data/processed_data/combined.json')
        
        # Process each entry
        logger.info(f"Processing {len(data)} entries...")
        for entry in tqdm(data, desc="Processing entries"):
            try:
                db._process_entry(entry)
                
            except EmbeddingError as e:
                logger.error(f"Error processing entry: {str(e)}")
                continue  # Skip to next entry if there's an error
                
        logger.info("Data processing completed successfully!")
        
        # Optional: Test some searches
        test_queries = [
            "楽天モバイル料金プラン",
            "iPhone",
            "お客様サポート"
        ]
        
        logger.info("\nTesting similarity searches...")
        for query in test_queries:
            logger.info(f"\nSearch query: '{query}'")
            results = db.find_most_similar(query, top_k=3)
            
            logger.info("Results:")
            for i, result in enumerate(results, 1):
                logger.info(f"{i}. Text: {result['text'][:100]}...")
                logger.info(f"   URL: {result['url']}")
                logger.info(f"   Similarity Score: {result['similarity']:.4f}")

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise
    
    finally:
        if 'db' in locals():
            del db  # Ensure database connection is closed

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    main()
