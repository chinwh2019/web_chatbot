import random
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .embeddings import EmbeddingDatabase
from .settings import settings
from .utils.exceptions import ChatbotError
from .utils.logger import setup_logger

logger = setup_logger("faq_bot", "faq_bot.log")


class ConversationTemplates:
    """Manages conversation templates for different intents."""

    def __init__(self):
        self.templates = {
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

    def get_response(self, intent: str) -> str:
        """Get a random response template for the given intent."""
        templates = self.templates.get(intent, self.templates["off_topic"])
        return random.choice(templates)


class IntentClassifier:
    """Classifies user queries into different intents."""

    def __init__(self):
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
        """Classify the user query into appropriate intent."""
        for intent, keywords in self.conversation_intents.items():
            if any(keyword in query for keyword in keywords):
                return {
                    "intent_type": "conversational",
                    "intent": intent,
                    "requires_search": False,
                }
        return {"intent_type": "business", "requires_search": True}


class FAQBot:
    """Main FAQ bot class handling user queries and responses."""

    def __init__(self, llm_client: OpenAI):
        """Initialize FAQ bot with necessary components."""
        try:
            self.embedding_db = EmbeddingDatabase()
            self.intent_classifier = IntentClassifier()
            self.conversation_templates = ConversationTemplates()
            self.llm_client = llm_client
            self.conversation_history: List[Dict[str, str]] = []
            logger.info("FAQ Bot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FAQ Bot: {str(e)}")
            raise ChatbotError(f"Bot initialization failed: {str(e)}")

    async def process_query(self, user_query: str) -> str:
        """Process user query and return appropriate response."""
        try:
            logger.info(f"Processing user query: {user_query}")
            self._update_conversation_history("user", user_query)

            intent_info = self.intent_classifier.classify(user_query)

            if not intent_info["requires_search"]:
                return self._handle_conversational_query(intent_info["intent"])

            return await self._handle_business_query(user_query)

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise ChatbotError(f"Query processing failed: {str(e)}")

    async def _handle_business_query(self, query: str) -> str:
        """Handle business-related queries using knowledge base."""
        try:
            relevant_info = self.embedding_db.find_most_similar(
                query, top_k=settings.TOP_K_RESULTS
            )
            prompt = self._prepare_prompt(query, relevant_info)
            llm_response = await self._get_llm_response(prompt)

            formatted_response = self._format_response(llm_response, relevant_info)
            self._update_conversation_history("assistant", formatted_response)

            return formatted_response

        except Exception as e:
            logger.error(f"Error handling business query: {str(e)}")
            raise ChatbotError(f"Failed to process business query: {str(e)}")

    def _handle_conversational_query(self, intent: str) -> str:
        """Handle conversational queries using templates."""
        response = self.conversation_templates.get_response(intent)
        self._update_conversation_history("assistant", response)
        return response

    async def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM."""
        try:
            logger.debug(f"Sending prompt to LLM: {prompt}")
            response = self.llm_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides information about Rakuten Mobile services in Japanese.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM response error: {str(e)}")
            raise ChatbotError(f"Failed to get LLM response: {str(e)}")

    def _prepare_prompt(self, query: str, relevant_info: List[Dict[str, Any]]) -> str:
        """Prepare prompt for LLM with context and relevant information."""
        relevant_text = "\n".join(
            f"• {info['text']} (Similarity: {info['similarity']:.2f})"
            for info in relevant_info
        )

        return f"""
        [Context]
        User Query: {query}
        
        [Relevant Information]
        {relevant_text}
        
        [Previous Conversation]
        {self._format_conversation_history()}
        
        Please provide a helpful response in Japanese that:
        1. Directly addresses the user's question
        2. Uses formal and polite language (です/ます調)
        3. Only includes information in the relevant information section, if applicable else provide a general response
        4. Maintains a professional tone
        """

    def _format_response(
        self, response: str, relevant_info: List[Dict[str, Any]]
    ) -> str:
        """Format the final response with references."""
        formatted = f"{response}\n\n"
        if relevant_info:
            formatted += "参考情報:\n"
            formatted += "\n".join(
                f"• {info['text']}: {info['url']}"
                for info in relevant_info
                if info["url"]
            )
        return formatted

    def _update_conversation_history(self, role: str, content: str) -> None:
        """Update conversation history with new message."""
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > 6:  # Keep last 3 exchanges
            self.conversation_history.pop(0)

    def _format_conversation_history(self) -> str:
        """Format conversation history for prompt context."""
        if not self.conversation_history:
            return "No previous conversation"
        return "\n".join(
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in self.conversation_history[-3:]
        )
