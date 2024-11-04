import asyncio
from openai import OpenAI
from typing import Optional
from src.settings import settings
from src.chatbot import FAQBot
from src.scraper import WebScraper
from src.data_processor import GPTContentProcessor
from src.embeddings import EmbeddingDatabase
from src.utils.exceptions import ChatbotError, WebScraperError
from src.utils.logger import logger


class WorkflowManager:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.scraper = WebScraper(output_dir=settings.SCRAPED_DATA_DIR)
        self.content_processor = GPTContentProcessor(api_key=settings.OPENAI_API_KEY)
        self.embedding_db = EmbeddingDatabase()
        self.chatbot: Optional[FAQBot] = None
        self._is_initialized: bool = False
        
        self._validate_config()

    def _validate_config(self) -> None:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not configured")
        if not settings.SCRAPED_DATA_DIR:
            raise ValueError("Scraped data directory is not configured")

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    async def initialize_chatbot_only(self) -> None:
        try:
            logger.info("Initializing chatbot with existing database data")
            self.chatbot = FAQBot(llm_client=self.client)
            self._is_initialized = True
            logger.info("✓ Chatbot initialized successfully")
        except Exception as e:
            self._is_initialized = False
            logger.error(f"Error initializing chatbot: {str(e)}")
            raise ChatbotError(f"Failed to initialize chatbot: {str(e)}") from e

    async def initialize_from_url(self, url: str) -> None:
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
            self._is_initialized = True
            logger.info("✓ Chatbot initialized successfully")

        except Exception as e:
            logger.error(f"Error in initialization: {str(e)}")
            raise

    async def chat_loop(self) -> None:
        if not self.is_initialized or not self.chatbot:
            raise ChatbotError("Chatbot not initialized")

        print("\nChat initialized! Type 'quit' to exit.")
        try:
            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['quit', 'exit']:
                    break

                response = await self.chatbot.process_query(user_input)
                print(f"\nAssistant: {response}")
        except KeyboardInterrupt:
            logger.info("Chat session terminated by user")
        except Exception as e:
            logger.error(f"Error in chat loop: {str(e)}")
            raise ChatbotError(f"Chat loop error: {str(e)}") from e

    async def shutdown(self) -> None:
        try:
            self._is_initialized = False
            logger.info("Workflow manager shut down successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            raise


async def main() -> None:
    workflow: Optional[WorkflowManager] = None
    
    try:
        workflow = WorkflowManager()
        
        while True:
            print("\n1. Scrape new URL")
            print("2. Use existing database")
            print("3. Quit")
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '3':
                break
            
            try:
                if choice == '1':
                    url = input("\nEnter the URL to scrape: ").strip()
                    print("\nInitializing system...")
                    url_full = settings.JINA_URL + url
                    await workflow.initialize_from_url(url_full)
                elif choice == '2':
                    print("\nInitializing chatbot with existing data...")
                    await workflow.initialize_chatbot_only()
                else:
                    print("\nInvalid choice. Please try again.")
                    continue
                
                await workflow.chat_loop()
                
                if input("\nWould you like to continue with another session? (y/n): ").strip().lower() != 'y':
                    break
                    
            except (ChatbotError, WebScraperError) as e:
                logger.error(f"Workflow error: {str(e)}")
                print(f"\nError: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"Error: {str(e)}")
    finally:
        if workflow:
            await workflow.shutdown()
        print("\nThank you for using the chatbot!")


if __name__ == "__main__":
    asyncio.run(main())