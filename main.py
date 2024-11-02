# from pathlib import Path
# from typing import Optional

# import asyncio
# from openai import OpenAI

# from scraper import Scrapper
# from data_processor import GPTContentProcessor
# from embeddings import EmbeddingDatabase
# from chatbot import EnhancedRakutenMobileFAQBot
# from config import api_key


# class WorkflowManager:
#     def __init__(self):
#         self.scraper = Scrapper('')
#         self.content_processor = GPTContentProcessor(api_key)
#         self.embedding_db = EmbeddingDatabase()
#         self.chatbot: Optional[EnhancedRakutenMobileFAQBot] = None

#         # Create necessary directories
#         Path("data").mkdir(exist_ok=True)
#         Path("processed_data").mkdir(exist_ok=True)
#         Path("embeddings").mkdir(exist_ok=True)

#     async def initialize_from_url(self, url: str) -> None:
#         """Initialize the entire pipeline from a URL"""
#         try:
#             # Step 1: Scrape the webpage
#             markdown_file = 'data/scraped_content.md'
#             self.scraper.fetch_and_save_webpage(url, markdown_file)
#             print("✓ Content scraped successfully")

#             # Step 2: Process the content
#             processed_data = self.content_processor.process_markdown_file(markdown_file)
#             if not processed_data:
#                 raise Exception("Failed to process content")
#             print("✓ Content processed successfully")

#             # Step 3: Generate and save embeddings
#             self.embedding_db.build_from_json('processed_data/all_processed_data.json')
#             self.embedding_db.save_database('embeddings/embeddings.pkl')
#             print("✓ Embeddings generated and saved")

#             # Step 4: Initialize chatbot
#             llm_client = OpenAI(api_key=api_key)
#             self.chatbot = EnhancedRakutenMobileFAQBot(
#                 llm_client=llm_client,
#                 embedding_db_path='embeddings/embeddings.pkl'
#             )
#             print("✓ Chatbot initialized successfully")

#         except Exception as e:
#             print(f"Error in initialization: {str(e)}")
#             raise

#     async def chat_loop(self):
#         """Interactive chat loop with the user"""
#         if not self.chatbot:
#             raise Exception("Chatbot not initialized. Please run initialize_from_url first.")

#         print("\nChat initialized! Type 'quit' to exit.")
#         while True:
#             try:
#                 user_input = input("\nYou: ").strip()
#                 if user_input.lower() in ['quit', 'exit']:
#                     break

#                 response = await self.chatbot.process_question(user_input)
#                 print(f"\nAssistant: {response}")

#             except Exception as e:
#                 print(f"Error processing message: {str(e)}")
#                 continue

# async def main():
#     # Initialize workflow manager
#     workflow = WorkflowManager()

#     # Get URL from user
#     url = input("Enter the URL to scrape: ")

#     # Initialize the system
#     print("\nInitializing system...")
#     await workflow.initialize_from_url(url)

#     # Start chat loop
#     await workflow.chat_loop()

# if __name__ == "__main__":
#     asyncio.run(main())

from openai import OpenAI

from src.chatbot import FAQBot
from src.settings import settings
from src.utils.exceptions import ChatbotError
from src.utils.logger import logger


async def main():
    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        bot = FAQBot(llm_client=client)

        # Example usage
        queries = [
            "こんにちは",
            "料金プランについて教えてください",
            "今日の天気はどうですか",
            "プラン変更の手続きを教えてください",
            "SIMカードの再発行手数料はいくらですか？",
            "口座振替の手数料について教えてください",
            "名義変更の手続き方法を教えてください",
            "tell me AU phone plans",
            "i am boring",
        ]

        for query in queries:
            response = await bot.process_query(query)
            print(response)

    except ChatbotError as e:
        logger.error(f"Bot error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
