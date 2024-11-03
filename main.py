from pathlib import Path
from typing import Optional
import asyncio
from openai import OpenAI

from src.scraper import WebScraper
from src.data_processor import GPTContentProcessor
from src.embeddings import EmbeddingDatabase
from src.chatbot import FAQBot
from src.settings import settings
from src.utils.exceptions import ChatbotError, WebScraperError
from src.utils.logger import logger


class WorkflowManager:
    """Manages the complete workflow from scraping to chatbot interaction."""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.scraper = WebScraper(output_dir=settings.SCRAPED_DATA_DIR)
        self.content_processor = GPTContentProcessor(api_key=settings.OPENAI_API_KEY)
        self.embedding_db = EmbeddingDatabase()
        self.chatbot: Optional[FAQBot] = None
        
        # Ensure all necessary directories exist
        self._setup_directories()
        
    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        settings.create_directories()
        
    async def initialize_from_url(self, url: str) -> None:
        """Initialize the entire pipeline from a URL."""
        try:
            # Step 1: Scrape the webpage
            logger.info(f"Scraping content from: {url}")
            markdown_file = self.scraper.scrape_and_save(url)
            logger.info("✓ Content scraped successfully")

            # Step 2: Process the content
            logger.info("Processing scraped content")
            processed_data = self.content_processor.process_markdown_file(markdown_file)
            if not processed_data:
                raise WebScraperError("Failed to process content")
            logger.info("✓ Content processed successfully")

            # Step 3: Store embeddings in PostgreSQL
            logger.info("Generating and storing embeddings")
            self.embedding_db._process_entry(processed_data)
            logger.info("✓ Embeddings generated and stored in database")

            # Step 4: Initialize chatbot
            logger.info("Initializing chatbot")
            self.chatbot = FAQBot(llm_client=self.client)
            logger.info("✓ Chatbot initialized successfully")

        except Exception as e:
            logger.error(f"Error in initialization: {str(e)}")
            raise

    async def chat_loop(self):
        """Interactive chat loop with the user."""
        if not self.chatbot:
            raise ChatbotError("Chatbot not initialized. Please run initialize_from_url first.")

        print("\nChat initialized! Type 'quit' to exit.")
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['quit', 'exit']:
                    break

                response = await self.chatbot.process_query(user_input)
                print(f"\nAssistant: {response}")

            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                print(f"\nError: {str(e)}")
                continue


async def main():
    try:
        # Initialize workflow manager
        workflow = WorkflowManager()
        
        while True:
            # Get URL from user
            url = input("\nEnter the URL to scrape (or 'quit' to exit): ").strip()
            
            if url.lower() == 'quit':
                break
                
            # Initialize the system with new URL
            print("\nInitializing system...")
            url_full = settings.JINA_URL + url
            await workflow.initialize_from_url(url_full)
            
            # Start chat loop
            await workflow.chat_loop()
            
            # Ask if user wants to scrape another URL
            again = input("\nWould you like to scrape another URL? (y/n): ").strip().lower()
            if again != 'y':
                break
                
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"Error: {str(e)}")
    finally:
        print("\nThank you for using the chatbot!")


if __name__ == "__main__":
    asyncio.run(main())