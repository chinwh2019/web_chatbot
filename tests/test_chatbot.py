import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.chatbot import ConversationTemplates, IntentClassifier, FAQBot

# Test data
MOCK_RELEVANT_INFO = [
    {
        "text": "Sample FAQ text 1",
        "url": "http://example.com/1",
        "similarity": 0.95
    },
    {
        "text": "Sample FAQ text 2",
        "url": "http://example.com/2",
        "similarity": 0.85
    }
]

MOCK_LLM_RESPONSE = "これは楽天モバイルのサービスについての回答です。"

@pytest.fixture
def conversation_templates():
    return ConversationTemplates()

@pytest.fixture
def intent_classifier():
    return IntentClassifier()

@pytest.fixture
def mock_llm_client():
    mock_client = Mock()
    mock_completion = Mock()
    mock_completion.choices = [Mock(message=Mock(content=MOCK_LLM_RESPONSE))]
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client

@pytest.fixture
def faq_bot(mock_llm_client):
    with patch('src.chatbot.EmbeddingDatabase') as mock_db:
        mock_db_instance = mock_db.return_value
        mock_db_instance.find_most_similar.return_value = MOCK_RELEVANT_INFO
        return FAQBot(mock_llm_client)

class TestConversationTemplates:
    def test_get_response_known_intent(self, conversation_templates):
        response = conversation_templates.get_response("greeting")
        assert isinstance(response, str)
        assert response in conversation_templates.templates["greeting"]

    def test_get_response_unknown_intent(self, conversation_templates):
        response = conversation_templates.get_response("unknown_intent")
        assert isinstance(response, str)
        assert response in conversation_templates.templates["off_topic"]

class TestIntentClassifier:
    @pytest.mark.parametrize("query,expected_intent", [
        ("こんにちは", "greeting"),
        ("さようなら", "farewell"),
        ("天気はどう", "small_talk"),
        ("映画が好き", "off_topic"),
        ("料金プランについて教えて", "business"),
    ])
    def test_classify_intents(self, intent_classifier, query, expected_intent):
        result = intent_classifier.classify(query)
        
        if expected_intent == "business":
            assert result["intent_type"] == "business"
            assert result["requires_search"] is True
        else:
            assert result["intent_type"] == "conversational"
            assert result["intent"] == expected_intent
            assert result["requires_search"] is False

class TestFAQBot:
    @pytest.mark.asyncio
    async def test_process_conversational_query(self, faq_bot):
        query = "こんにちは"
        response = await faq_bot.process_query(query)
        assert isinstance(response, str)
        assert len(faq_bot.conversation_history) == 2  # Query and response

    @pytest.mark.asyncio
    async def test_process_business_query(self, faq_bot):
        query = "料金プランについて教えて"
        response = await faq_bot.process_query(query)
        assert isinstance(response, str)
        assert MOCK_LLM_RESPONSE in response
        assert "詳細については" in response
        assert len(faq_bot.conversation_history) == 2

    def test_conversation_history_limit(self, faq_bot):
        # Add 8 messages (4 exchanges)
        for i in range(8):
            role = "user" if i % 2 == 0 else "assistant"
            faq_bot._update_conversation_history(role, f"Message {i}")
        
        assert len(faq_bot.conversation_history) == 6  # Should keep only last 6 messages

    @pytest.mark.asyncio
    async def test_error_handling(self, faq_bot, mock_llm_client):
        mock_llm_client.chat.completions.create.side_effect = Exception("API Error")
        
        with pytest.raises(Exception):
            await faq_bot.process_query("料金プランについて教えて")

    def test_format_response(self, faq_bot):
        response = faq_bot._format_response(MOCK_LLM_RESPONSE, MOCK_RELEVANT_INFO)
        assert MOCK_LLM_RESPONSE in response
        assert "詳細については" in response
        assert "http://example.com/1" in response
        assert "http://example.com/2" in response