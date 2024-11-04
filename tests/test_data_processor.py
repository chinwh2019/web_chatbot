import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import pytest
from unittest.mock import Mock, patch
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from src.data_processor import GPTContentProcessor, ContentValidator, FileHandler
from src.utils.exceptions import WebScraperError

# Test data
SAMPLE_MARKDOWN = """# Test Title
This is test content
[Test Link](http://test.com)
"""

SAMPLE_GPT_RESPONSE = {
    "title": "Test Title",
    "content": "This is test content",
    "links": [{"text": "Test Link", "url": "http://test.com"}]
}

@pytest.fixture
def mock_openai_response():
    mock_message = ChatCompletionMessage(
        content=json.dumps(SAMPLE_GPT_RESPONSE),
        role="assistant",
        function_call=None,
        tool_calls=None
    )
    mock_choice = Choice(
        finish_reason="stop",
        index=0,
        message=mock_message,
        logprobs=None
    )
    return ChatCompletion(
        id="test_id",
        choices=[mock_choice],
        created=1234567890,
        model="gpt-3.5-turbo",
        object="chat.completion",
        system_fingerprint=None,
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    )

class TestGPTContentProcessor:
    @pytest.fixture
    def processor(self):
        return GPTContentProcessor(api_key="test_key")

    def test_happy_path_process_markdown_file(self, processor, mock_openai_response, tmp_path):
        # Create a test markdown file
        test_file = tmp_path / "test.md"
        test_file.write_text(SAMPLE_MARKDOWN)

        # Mock OpenAI client response
        with patch.object(processor.client.chat.completions, 'create', return_value=mock_openai_response):
            result = processor.process_markdown_file(test_file)

        assert result is not None
        assert result["title"] == "Test Title"
        assert result["content"] == "This is test content"
        assert len(result["links"]) == 1
        assert result["links"][0]["text"] == "Test Link"
        assert result["links"][0]["url"] == "http://test.com"

    def test_error_invalid_api_key(self):
        with pytest.raises(ValueError, match="API key is required"):
            GPTContentProcessor(api_key="")

class TestContentValidator:
    def test_validate_json_structure_happy_path(self):
        validator = ContentValidator()
        valid_data = {
            "title": "Test",
            "content": "Content",
            "links": []
        }
        assert validator.validate_json_structure(valid_data) is True

    def test_validate_json_structure_missing_fields(self):
        validator = ContentValidator()
        invalid_data = {
            "title": "Test",
            "content": "Content"
            # missing 'links' field
        }
        assert validator.validate_json_structure(invalid_data) is False

class TestFileHandler:
    def test_read_markdown_file_happy_path(self, tmp_path):
        handler = FileHandler()
        test_file = tmp_path / "test.md"
        test_content = "# Test Content"
        test_file.write_text(test_content)

        result = handler.read_markdown_file(test_file)
        assert result == test_content

    def test_read_markdown_file_not_found(self):
        handler = FileHandler()
        with pytest.raises(WebScraperError, match="Failed to read file"):
            handler.read_markdown_file(Path("nonexistent_file.md"))
