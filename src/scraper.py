from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from html2text import HTML2Text

from .settings import settings
from .utils.exceptions import WebScraperError
from .utils.logger import logger


class WebScraper:
    """Handles web scraping operations with proper error handling and logging"""

    def __init__(self, output_dir: str | Path = "data"):
        self.output_dir = Path(output_dir)
        self._setup_directories()
        self.html2text = self._configure_html2text()
        self.input_file = None
        logger.info("WebScraper initialized successfully")

    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Output directory setup complete: {self.output_dir}")
        except Exception as e:
            logger.error(f"Failed to create directories: {str(e)}")
            raise WebScraperError(f"Directory setup failed: {str(e)}")

    @staticmethod
    def _configure_html2text() -> HTML2Text:
        """Configure HTML2Text converter with optimal settings"""
        h2t = HTML2Text()
        h2t.ignore_links = False
        h2t.ignore_images = False
        h2t.ignore_tables = False
        h2t.body_width = 0
        return h2t

    def _generate_filename(self, url: str) -> str:
        """Generate unique filename based on URL and timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain = urlparse(url).netloc.replace(".", "_")
        return f"{domain}_{timestamp}"

    def fetch_and_save_webpage(self, url: str, output_file: str) -> None:
        """
        Fetches webpage content and saves it as markdown

        Args:
            url: The webpage URL to fetch
            output_file: Path to save the markdown file
        """
        try:
            # Fetch webpage content
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Convert HTML to markdown
            markdown_content = self.html2text.handle(response.text)

            # Save to file
            output_path = Path(output_file)
            output_path.write_text(markdown_content, encoding="utf-8")

            self.input_file = output_file  # Update input file path
            logger.info(f"Successfully saved content to {output_file}")

        except requests.RequestException as e:
            logger.error(f"Failed to fetch webpage {url}: {str(e)}")
            raise WebScraperError(f"Failed to fetch webpage: {str(e)}")
        except IOError as e:
            logger.error(f"Failed to save markdown file: {str(e)}")
            raise WebScraperError(f"Failed to save markdown file: {str(e)}")

    def scrape_and_save(self, url: str) -> str:
        """Main method to scrape webpage and save content"""
        logger.info(f"Starting scrape operation for URL: {url}")

        try:
            # Generate unique filename
            filename = self._generate_filename(url)
            output_file = str(self.output_dir / f"{filename}.md")

            # Fetch and save webpage
            self.fetch_and_save_webpage(url, output_file)

            return output_file

        except WebScraperError as e:
            logger.error(f"Scrape operation failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during scrape operation: {str(e)}")
            raise WebScraperError(f"Scrape operation failed: {str(e)}")


def main():
    # Initialize the scraper with custom output directory (optional)
    scraper = WebScraper(output_dir=settings.SCRAPED_DATA_DIR)

    # List of URLs to scrape
    urls = [
        # "https://network.mobile.rakuten.co.jp/support/",
        "https://network.mobile.rakuten.co.jp/support/payment/bill/?l-id=support_top_member_category"
    ]

    # Scrape each URL
    for url in urls:
        try:
            url = "https://r.jina.ai/" + url
            # scrape_and_save returns a dictionary with paths to saved files
            result = scraper.scrape_and_save(url)

            print(f"Successfully scraped {url}")
            print(f"Markdown file saved at: {result}")
            print("-" * 50)

        except Exception as e:
            print(f"Failed to scrape {url}: {str(e)}")


if __name__ == "__main__":
    main()
