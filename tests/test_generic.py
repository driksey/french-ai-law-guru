"""
Generic integration tests for the French AI Law Assistant.

These tests focus on overall functionality rather than individual functions,
making them more resilient to code changes.
"""

import tempfile
import pytest
from unittest.mock import Mock, patch

from legal_ai_assistant.config import LLM_CONFIG, EMBEDDING_CONFIG, APP_CONFIG
from legal_ai_assistant.chat_handler import process_question_with_agent
from langchain_core.messages import HumanMessage, AIMessage


class TestConfiguration:
    """Test that configuration is valid and complete."""

    def test_config_structure(self):
        """Test that all required configuration sections exist."""
        # Test that main config sections exist
        assert "LLM_CONFIG" in globals()
        assert "EMBEDDING_CONFIG" in globals()
        assert "APP_CONFIG" in globals()

        # Test that configs have required keys
        required_llm_keys = ["model_name", "max_new_tokens", "context_window"]
        required_embedding_keys = ["model_name", "batch_size", "device"]
        required_app_keys = ["title", "max_top_k", "default_top_k"]

        for key in required_llm_keys:
            assert key in LLM_CONFIG
        for key in required_embedding_keys:
            assert key in EMBEDDING_CONFIG
        for key in required_app_keys:
            assert key in APP_CONFIG

    def test_config_values_valid(self):
        """Test that configuration values are within reasonable ranges."""
        # LLM config validation
        assert LLM_CONFIG["max_new_tokens"] > 0
        assert LLM_CONFIG["context_window"] > 0
        assert 0 <= LLM_CONFIG["temperature"] <= 2
        assert 0 <= LLM_CONFIG["top_p"] <= 1

        # Embedding config validation
        assert EMBEDDING_CONFIG["batch_size"] > 0
        assert EMBEDDING_CONFIG["device"] in ["cpu", "cuda"]

        # App config validation
        assert APP_CONFIG["max_top_k"] > 0
        assert APP_CONFIG["default_top_k"] > 0
        assert APP_CONFIG["default_top_k"] <= APP_CONFIG["max_top_k"]


class TestCoreFunctionality:
    """Test core application functionality."""

    def test_chat_handler_basic_functionality(self):
        """Test that chat handler can process questions without errors."""
        # Mock agent that returns a simple response
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Test response")]
        }

        with patch("langdetect.detect", return_value="en"):
            result = process_question_with_agent(mock_agent, "test question")
            answer, was_rag_used, retrieved_docs = result

            # Basic validation
            assert isinstance(answer, str)
            assert len(answer) > 0
            assert isinstance(was_rag_used, bool)
            assert isinstance(retrieved_docs, list)

    def test_chat_handler_error_handling(self):
        """Test that chat handler handles errors gracefully."""
        # Mock agent that raises an exception
        mock_agent = Mock()
        mock_agent.invoke.side_effect = Exception("Test error")

        with patch("langdetect.detect", side_effect=Exception("Detection error")):
            result = process_question_with_agent(mock_agent, "test question")
            answer, was_rag_used, retrieved_docs = result

            # Should handle errors gracefully
            assert isinstance(answer, str)
            assert "Sorry" in answer or "error" in answer.lower()
            assert was_rag_used is False
            assert isinstance(retrieved_docs, list)


class TestDocumentProcessing:
    """Test document processing functionality."""

    def test_pdf_loading_empty_directory(self):
        """Test PDF loading from empty directory."""
        from legal_ai_assistant.utils import load_pdfs

        with tempfile.TemporaryDirectory() as temp_dir:
            pdfs = load_pdfs(temp_dir)
            assert isinstance(pdfs, list)
            assert len(pdfs) == 0

    def test_document_preprocessing(self):
        """Test document preprocessing with mock data."""
        from legal_ai_assistant.utils import preprocess_pdfs

        # Create mock documents
        class MockDoc:
            def __init__(self, content, source):
                self.page_content = content
                self.metadata = {"source": source}

        mock_docs = [
            MockDoc("Test content 1", "legal_docs/test1.pdf"),
            MockDoc("Test content 2", "legal_docs/test2.pdf"),
        ]

        processed_docs = preprocess_pdfs(mock_docs)

        # Basic validation
        assert isinstance(processed_docs, list)
        assert len(processed_docs) > 0
        for doc in processed_docs:
            assert hasattr(doc, "metadata")
            assert "source" in doc.metadata
            assert "category" in doc.metadata


class TestAgentFunctionality:
    """Test agent-related functionality."""

    def test_prompt_creation(self):
        """Test that prompts can be created for different languages."""
        from legal_ai_assistant.agents import create_prompt_strict

        # Test with different languages
        languages = ["en", "fr", "es", None]

        for lang in languages:
            prompt = create_prompt_strict(lang)
            assert prompt is not None
            assert hasattr(prompt, "format") or hasattr(prompt, "invoke")

    def test_tool_call_parsing(self):
        """Test tool call parsing with various formats."""
        from legal_ai_assistant.agents import parse_tool_call

        # Test different JSON formats
        test_cases = [
            '{"name": "tool_rag", "arguments": {"query": "test"}}',
            '```json\n{"name": "tool_rag", "arguments": {"query": "test"}}\n```',
            '```\n{"name": "tool_rag", "arguments": {"query": "test"}}\n```',
        ]

        for test_case in test_cases:
            result = parse_tool_call(test_case)
            assert isinstance(result, list)
            assert len(result) > 0

    def test_rag_tool_creation(self):
        """Test RAG tool creation."""
        from legal_ai_assistant.agents import create_rag_tool

        # Mock retriever
        mock_retriever = Mock()

        tool = create_rag_tool(mock_retriever)
        assert tool.name == "tool_rag"
        assert "legal documents" in tool.description.lower()


class TestTokenCalculation:
    """Test token calculation functionality."""

    def test_token_estimation(self):
        """Test token estimation with various inputs."""
        from legal_ai_assistant.utils import estimate_tokens_from_chars

        test_cases = [
            ("", 0),
            ("test", 1),
            ("hello world", 3),
            ("a" * 100, 31),
        ]

        for text, expected_min in test_cases:
            result = estimate_tokens_from_chars(text)
            assert isinstance(result, int)
            assert result >= expected_min

    def test_max_response_tokens_calculation(self):
        """Test max response tokens calculation."""
        from legal_ai_assistant.utils import calculate_max_response_tokens

        # Test with different content lengths
        test_cases = [
            ("Short content", "Short question"),
            ("Medium content with more text", "Medium question"),
            ("Very long content with many characters", "Very long question"),
        ]

        for content, question in test_cases:
            result = calculate_max_response_tokens(content, question)
            assert isinstance(result, int)
            assert result > 0
            assert result <= LLM_CONFIG["max_new_tokens"]


class TestIntegration:
    """Integration tests that test multiple components together."""

    def test_end_to_end_question_processing(self):
        """Test complete question processing workflow."""
        # Mock agent with realistic response
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "messages": [
                AIMessage(content="Based on the legal documents, here is the answer...")
            ]
        }

        with patch("langdetect.detect", return_value="en"):
            result = process_question_with_agent(
                mock_agent, "What are the GDPR requirements for AI systems?"
            )
            answer, was_rag_used, retrieved_docs = result

            # Validate complete response
            assert isinstance(answer, str)
            assert len(answer) > 10
            assert isinstance(was_rag_used, bool)
            assert isinstance(retrieved_docs, list)

            # Check that timing info is included
            assert "seconds" in answer

    def test_multilingual_support(self):
        """Test multilingual question processing."""
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Réponse en français")]
        }

        languages = ["en", "fr", "es"]

        for lang in languages:
            with patch("langdetect.detect", return_value=lang):
                result = process_question_with_agent(mock_agent, "Test question")
                answer, was_rag_used, retrieved_docs = result

                assert isinstance(answer, str)
                assert len(answer) > 0
                assert isinstance(was_rag_used, bool)
                assert isinstance(retrieved_docs, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
