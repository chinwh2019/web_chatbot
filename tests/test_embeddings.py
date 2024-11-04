import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.embeddings import EmbeddingDatabase
from src.models import Document

@pytest.fixture
def mock_db():
    with patch('src.embeddings.OpenAI'), \
         patch('src.embeddings.create_engine'), \
         patch('src.embeddings.sessionmaker'):
        db = EmbeddingDatabase()
        # Create a mock session
        db.session = Mock()
        return db

@pytest.fixture
def sample_embedding():
    return np.array([0.1, 0.2, 0.3])  # Simplified embedding for testing

def test_process_entry(mock_db, sample_embedding):
    # Mock generate_embedding to return our sample embedding
    mock_db.generate_embedding = Mock(return_value=sample_embedding)
    
    # Test entry with all possible fields
    test_entry = {
        "title": "Test Title",
        "content": "Test Content",
        "links": [
            {"text": "Link 1", "url": "http://example1.com"},
            {"text": "Link 2", "url": "http://example2.com"}
        ]
    }
    
    # Process the entry
    mock_db._process_entry(test_entry)
    
    # Verify bulk_save_objects was called with correct documents
    calls = mock_db.session.bulk_save_objects.call_args[0][0]
    
    # Check if we got 4 documents (1 title + 1 content + 2 links)
    assert len(calls) == 4
    
    # Verify the documents have correct attributes
    title_doc = next(doc for doc in calls if doc.doc_type == "title")
    content_doc = next(doc for doc in calls if doc.doc_type == "content")
    link_docs = [doc for doc in calls if doc.doc_type == "link"]
    
    assert title_doc.text == "Test Title"
    assert content_doc.text == "Test Content"
    assert len(link_docs) == 2
    assert link_docs[0].url == "http://example1.com"
    assert link_docs[1].url == "http://example2.com"

def test_find_most_similar(mock_db, sample_embedding):
    # Mock generate_embedding
    mock_db.generate_embedding = Mock(return_value=sample_embedding)
    
    # Create mock documents with known embeddings
    mock_docs = [
        Document(
            text="Doc 1",
            url="url1",
            embedding=np.array([0.1, 0.2, 0.3]),  # Same as query, should be most similar
            doc_type="content"
        ),
        Document(
            text="Doc 2",
            url="url2",
            embedding=np.array([-0.1, -0.2, -0.3]),  # Opposite direction, least similar
            doc_type="content"
        ),
        Document(
            text="Doc 3",
            url="url3",
            embedding=np.array([0.05, 0.15, 0.25]),  # Changed values for different similarity
            doc_type="content"
        )
    ]
    
    # Mock the query results
    mock_db.session.query.return_value.all.return_value = mock_docs
    
    # Test the find_most_similar method
    results = mock_db.find_most_similar("test query", top_k=2)
    
    # Verify results
    assert len(results) == 2
    assert results[0]["text"] == "Doc 1"  # Should be most similar
    assert results[0]["similarity"] == 1.0  # Perfect similarity
    assert results[1]["text"] == "Doc 3"  # Should be second most similar
    assert 0 < results[1]["similarity"] < 1  # Partial similarity

def test_process_entry_error_handling(mock_db):
    mock_db.generate_embedding = Mock(side_effect=Exception("Test error"))
    
    with pytest.raises(Exception):
        mock_db._process_entry({"title": "Test"})
    
    # Verify session rollback was called
    mock_db.session.rollback.assert_called_once()

def test_find_most_similar_error_handling(mock_db):
    mock_db.generate_embedding = Mock(side_effect=Exception("Test error"))
    
    with pytest.raises(Exception):
        mock_db.find_most_similar("test query")
