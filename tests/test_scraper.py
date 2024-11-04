import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from requests.exceptions import RequestException

from src.scraper import WebScraper
from src.utils.exceptions import WebScraperError

@pytest.fixture
def web_scraper(tmp_path):
    """Fixture to create WebScraper instance with temporary directory"""
    return WebScraper(output_dir=tmp_path)

@pytest.fixture
def mock_response():
    """Fixture to create a mock response"""
    mock = Mock()
    mock.text = "<html><body><h1>Test Content</h1></body></html>"
    mock.raise_for_status.return_value = None
    return mock

def test_scrape_and_save_happy_path(web_scraper, mock_response):
    """Test successful webpage scraping and saving"""
    test_url = "https://example.com"
    
    # Mock the requests.get call
    with patch('requests.get', return_value=mock_response):
        output_file = web_scraper.scrape_and_save(test_url)
        
        # Verify the output file exists
        assert Path(output_file).exists()
        
        # Verify the content was saved
        content = Path(output_file).read_text(encoding='utf-8')
        assert "Test Content" in content
        
        # Verify the file has .md extension
        assert output_file.endswith('.md')

def test_scrape_and_save_request_error(web_scraper):
    """Test handling of request error during scraping"""
    test_url = "https://nonexistent-example.com"
    
    # Mock requests.get to raise an exception
    with patch('requests.get', side_effect=RequestException("Failed to fetch")):
        with pytest.raises(WebScraperError) as exc_info:
            web_scraper.scrape_and_save(test_url)
        
        assert "Failed to fetch webpage" in str(exc_info.value)