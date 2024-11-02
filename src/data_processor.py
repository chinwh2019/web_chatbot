import hashlib
import json
import logging  # Add this import
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI
from tenacity import (after_log, before_log, retry, retry_if_exception_type,
                      stop_after_attempt, wait_exponential)

from settings import settings
from utils.exceptions import WebScraperError
from utils.logger import logger


class ContentValidator:
    """Validate and clean content data"""

    @staticmethod
    def validate_json_structure(data: Dict[str, Any]) -> bool:
        required_fields = {"title", "content", "links"}
        return all(field in data for field in required_fields)

    @staticmethod
    def clean_json_string(json_str: str) -> str:
        """Clean JSON string from markdown formatting."""
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            lines = [line for line in lines if not line.startswith("```")]
            return "\n".join(lines)
        return json_str


class FileHandler:
    """Handle file operations"""

    @staticmethod
    def generate_unique_filename(base_path: Path, content: str) -> Path:
        """Generate unique filename based on content hash and timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return base_path / f"{timestamp}_{content_hash}.json"

    @staticmethod
    def read_markdown_file(file_path: Path) -> str:
        """Read content from markdown file."""
        try:
            logger.debug(f"Reading file: {file_path}")
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise WebScraperError(f"Failed to read file {file_path}: {str(e)}")

    @staticmethod
    def save_json_data(file_path: Path, data: Dict[str, Any]) -> None:
        """Save JSON data to file."""
        try:
            file_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info(f"Saved data to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise WebScraperError(f"Failed to save data: {str(e)}")


class GPTContentProcessor:
    """Process markdown content using GPT models with structured output."""

    def __init__(self, api_key: str):
        self._validate_api_key(api_key)
        self.client = OpenAI(api_key=api_key)
        self.file_handler = FileHandler()
        self.content_validator = ContentValidator()
        self._setup_output_directory()
        logger.info("GPTContentProcessor initialized successfully")

    @staticmethod
    def _validate_api_key(api_key: str) -> None:
        if not api_key:
            logger.error("API key not provided")
            raise ValueError("API key is required")

    def _setup_output_directory(self) -> None:
        """Create output directory if it doesn't exist"""
        settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output directory setup complete: {settings.PROCESSED_DATA_DIR}")

    def _get_system_prompt(self) -> str:
        """Get system prompt for GPT."""
        return """
        You are a content structuring assistant. Convert the provided markdown web content into a structured JSON format.
        Follow these rules:
        1. Extract main sections and subsections
        2. Preserve hierarchical relationships
        3. Extract all links with their text and URLs
        4. Identify navigation elements
        5. Separate content into logical chunks
        
        Use this JSON structure:
        {
            "title": "page title",
            "content": "main content text if available",
            "links": [
                {
                    "text": "link text and description if available",
                    "url": "link url",
                }
            ]
        }
        """

    @retry(
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(multiplier=settings.RETRY_DELAY, min=4, max=10),
        retry=retry_if_exception_type((Exception)),
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.DEBUG),
    )
    def process_content(self, markdown_content: str) -> Dict[str, Any]:
        """Process markdown content using GPT model with retry mechanism."""
        try:
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {
                        "role": "user",
                        "content": f"Please convert this markdown content into structured JSON:\n{markdown_content}",
                    },
                ],
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
            )

            return self._parse_gpt_response(response)

        except Exception as e:
            logger.error(f"Error processing content: {str(e)}")
            raise WebScraperError(f"Failed to process content: {str(e)}")

    def _parse_gpt_response(self, response: Any) -> Dict[str, Any]:
        """Parse and validate GPT response."""
        try:
            json_content = response.choices[0].message.content
            json_content = self.content_validator.clean_json_string(json_content)
            structured_content = json.loads(json_content)

            if not self.content_validator.validate_json_structure(structured_content):
                raise WebScraperError("Invalid JSON structure in GPT response")

            logger.debug("Successfully parsed GPT response")
            return structured_content

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error(f"Raw response content: {response.choices[0].message.content}")
            raise WebScraperError(f"Failed to parse JSON response: {str(e)}")

    def process_markdown_file(
        self, file_path: Union[str, Path]
    ) -> Optional[Dict[str, Any]]:
        """Process a markdown file and return structured content."""
        file_path = Path(file_path)
        try:
            content = self.file_handler.read_markdown_file(file_path)
            structured_data = self.process_content(content)

            # Generate unique output path
            output_path = self.file_handler.generate_unique_filename(
                settings.PROCESSED_DATA_DIR, content
            )

            self.file_handler.save_json_data(output_path, structured_data)
            logger.info(f"Successfully processed file: {file_path}")
            return structured_data

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None

    def batch_process_files(
        self, file_paths: List[Union[str, Path]], max_workers: int = 4
    ) -> List[Dict[str, Any]]:
        """Process multiple markdown files in parallel."""
        processed_files = []
        total_files = len(file_paths)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_markdown_file, file_path): file_path
                for file_path in file_paths
            }

            for idx, future in enumerate(future_to_file, 1):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        processed_files.append(result)
                        logger.info(
                            f"Successfully processed ({idx}/{total_files}): {file_path}"
                        )
                    else:
                        logger.warning(
                            f"Failed to process ({idx}/{total_files}): {file_path}"
                        )
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")

        return processed_files


def main():
    """Main entry point for the script"""
    try:
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        processor = GPTContentProcessor(api_key=api_key)

        # Process all markdown files in the data directory
        markdown_files = list(settings.SCRAPED_DATA_DIR.glob("*.md"))

        if not markdown_files:
            logger.warning("No markdown files found in the data directory")
            return

        processed_data = processor.batch_process_files(markdown_files)

        # Save combined results with unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_output = (
            settings.PROCESSED_DATA_DIR / f"combined_data_{timestamp}.json"
        )
        FileHandler.save_json_data(combined_output, processed_data)
        logger.info(f"Processing complete. Combined results saved to {combined_output}")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
