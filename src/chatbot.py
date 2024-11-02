import random
from typing import Any, Dict, List

import numpy as np
import openai
from dotenv import load_dotenv
from openai import OpenAI

from embeddings import EmbeddingDatabase


class ResponseTemplates:
    def __init__(self):
        self.conversation_templates = {
            "greeting": [
                "こんにちは！楽天モバイルのアシスタントです。ご用件をお聞かせください。",
                "いつもご利用ありがとうございます。楽天モバイルのサポートです。",
            ],
            "farewell": [
                "ご利用ありがとうございました。また何かございましたらお気軽にお声がけください。",
                "お気をつけてお過ごしください。また何かございましたらご相談ください。",
            ],
            "small_talk": [
                "ご親切にありがとうございます。楽天モバイルに関することでお困りの点がございましたら、お気軽にお申し付けください。"
            ],
            "off_topic": [
                "申し訳ございませんが、楽天モバイルのサービスに関する質問にのみ回答させていただいております。",
                "楽天モバイルのサービスについて、お困りの点がございましたらお申し付けください。",
            ],
        }

    def get_conversation_response(self, intent: str) -> str:
        templates = self.conversation_templates.get(
            intent, self.conversation_templates["off_topic"]
        )
        return random.choice(templates)


class IntentClassifier:
    def __init__(self):
        # Separate intents into business-related and conversational
        self.conversation_intents = {
            "greeting": [
                "こんにちは",
                "おはよう",
                "こんばんは",
                "はじめまして",
                "よろしく",
            ],
            "farewell": ["さようなら", "ありがとう", "お疲れ様", "また"],
            "small_talk": ["天気", "調子", "元気", "どう"],
            "off_topic": ["映画", "音楽", "食事", "スポーツ"],
        }

    def classify(self, query: str) -> Dict[str, Any]:
        # Check for conversation intents first
        for intent, keywords in self.conversation_intents.items():
            if any(keyword in query for keyword in keywords):
                return {
                    "intent_type": "conversational",
                    "intent": intent,
                    "requires_search": False,
                }

        # If not conversational, it's a business query
        return {"intent_type": "business", "requires_search": True}


class EnhancedRakutenMobileFAQBot:
    def __init__(self, llm_client: OpenAI, embedding_db_path: str):
        self.embedding_db = EmbeddingDatabase()
        self.embedding_db.load_database(embedding_db_path)
        self.intent_classifier = IntentClassifier()
        self.response_templates = ResponseTemplates()
        self.llm_client = llm_client
        self.conversation_history: List[Dict[str, str]] = []

    async def process_question(self, user_query: str) -> str:
        try:
            # Store conversation history
            self.conversation_history.append({"role": "user", "content": user_query})

            # Classify intent
            intent_info = self.intent_classifier.classify(user_query)

            # Handle conversational queries directly with templates
            if not intent_info["requires_search"]:
                response = self.response_templates.get_conversation_response(
                    intent_info["intent"]
                )
                self.conversation_history.append(
                    {"role": "assistant", "content": response}
                )
                return response

            # For business queries, perform full search and LLM processing
            relevant_info = self.embedding_db.find_most_similar(user_query, top_k=3)
            prompt = self.prepare_prompt(user_query, relevant_info)
            llm_response = await self.get_llm_response(prompt)
            formatted_response = self.format_response(llm_response, relevant_info)

            # Store bot response in history
            self.conversation_history.append(
                {"role": "assistant", "content": formatted_response}
            )
            return formatted_response

        except Exception as e:
            return self.handle_error(e)

    def prepare_prompt(self, query: str, relevant_info: List[Dict[str, Any]]) -> str:
        relevant_text = "\n".join(
            [
                f"• {info['text']} (Similarity: {info['similarity']:.2f})"
                for info in relevant_info
            ]
        )

        return f"""
        [Context]
        User Query: {query}
        
        [Relevant Information]
        {relevant_text}
        
        [Previous Conversation]
        {self.format_conversation_history()}
        
        Please provide a helpful response in Japanese that:
        1. Directly addresses the user's question
        2. Uses formal and polite language (です/ます調)
        3. Only includes information in the relevant information section, if applicable else provide a general response
        4. Maintains a professional tone
        """

    async def get_llm_response(self, prompt: str) -> str:
        try:
            print(f"LLM Prompt: {prompt}")
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides information about Rakuten Mobile services in Japanese.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=400,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in LLM response: {str(e)}")
            return "申し訳ございませんが、現在システムエラーが発生しています。"

    def format_response(
        self, response: str, relevant_info: List[Dict[str, Any]]
    ) -> str:
        formatted = f"{response}\n\n"
        if relevant_info:
            formatted += "参考情報:\n"
            for info in relevant_info:
                if info["url"]:  # Only add reference if URL exists
                    formatted += f"• {info['text']}: {info['url']}\n"
        return formatted

    def format_conversation_history(self) -> str:
        if not self.conversation_history:
            return "No previous conversation"

        formatted_history = []
        for msg in self.conversation_history[-3:]:  # Only include last 3 messages
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history.append(f"{role}: {msg['content']}")

        return "\n".join(formatted_history)

    def handle_error(self, error: Exception) -> str:
        error_msg = f"エラーが発生しました: {str(error)}"
        print(f"Error in processing: {error}")  # For logging
        return error_msg


async def main():
    # Initialize the bot
    api_key = load_dotenv()  # Load variables from .env
    openai.api_key = api_key
    llm_client = OpenAI()
    bot = EnhancedRakutenMobileFAQBot(
        llm_client=llm_client, embedding_db_path="embeddings/embeddings.pkl"
    )

    try:
        # Example questions
        questions = [
            "こんにちは",  # Greeting -> Direct template response
            "料金プランを教えてください",  # Business query -> Full search and LLM
            "今日の天気はどうですか",  # Off-topic -> Template response
            "プラン変更の手続きを教えてください"  # Business query -> Full search and LLM
            "SIMカードの再発行手数料はいくらですか？",
            "口座振替の手数料について教えてください",
            "名義変更の手続き方法を教えてください",
            "tell me AU phone plans",
            "i am boring",
        ]

        for question in questions:
            print(f"\nQ: {question}")
            response = await bot.process_question(question)
            print(f"A: {response}")

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
