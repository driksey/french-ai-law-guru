import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from legal_ai_assistant.agents import (
    BasicToolNode,
    create_final_answer,
    route_tools
)
from legal_ai_assistant.utils import (
    retrieve_documents,
    store_docs_with_embeddings,
    load_or_create_vectorstore
)


class TestBasicToolNode:
    """Test suite for BasicToolNode."""
    
    def test_basic_tool_node_init(self):
        """Test BasicToolNode initialization."""
        tools = [Mock()]
        node = BasicToolNode(tools)
        assert len(node.tools_by_name) == 1

    def test_basic_tool_node_call_success(self):
        """Test successful tool execution."""
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.invoke.return_value = "tool_result"
        
        node = BasicToolNode([mock_tool])
        
        inputs = {
            "messages": [
                AIMessage(content="test", tool_calls=[{
                    "name": "test_tool",
                    "args": {"param": "value"},
                    "id": "test_id",
                    "type": "function"
                }])
            ]
        }
        
        result = node(inputs)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "tool_result"

    def test_basic_tool_node_call_error(self):
        """Test tool execution with error."""
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.invoke.side_effect = Exception("Tool error")
        
        node = BasicToolNode([mock_tool])
        
        inputs = {
            "messages": [
                AIMessage(content="test", tool_calls=[{
                    "name": "test_tool",
                    "args": {"param": "value"},
                    "id": "test_id",
                    "type": "function"
                }])
            ]
        }
        
        result = node(inputs)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "[ERROR]" in result["messages"][0].content

    def test_basic_tool_node_no_messages(self):
        """Test BasicToolNode with no messages."""
        node = BasicToolNode([])
        
        with pytest.raises(ValueError):
            node({})

    def test_basic_tool_node_unknown_tool(self):
        """Test BasicToolNode with unknown tool."""
        node = BasicToolNode([])
        
        inputs = {
            "messages": [
                AIMessage(content="test", tool_calls=[{
                    "name": "unknown_tool",
                    "args": {"param": "value"},
                    "id": "test_id"
                }])
            ]
        }
        
        with pytest.raises(ValueError):
            node(inputs)


class TestCreateFinalAnswer:
    """Test suite for create_final_answer function."""
    
    @patch('legal_ai_assistant.agents.detect')
    @patch('legal_ai_assistant.agents.calculate_max_response_tokens')
    def test_create_final_answer_success(self, mock_calc_tokens, mock_detect):
        """Test successful final answer creation."""
        mock_detect.return_value = "en"
        mock_calc_tokens.return_value = 200
        
        mock_chat_model = Mock()
        mock_chat_model.invoke.return_value = Mock(content="Final answer")
        
        state = {
            "messages": [
                HumanMessage(content="User question"),
                ToolMessage(content="Tool result", tool_call_id="parsed-tool-call")
            ]
        }
        
        result = create_final_answer(state, mock_chat_model)
        
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "Final answer"

    def test_create_final_answer_no_tool_messages(self):
        """Test final answer creation with no tool messages."""
        mock_chat_model = Mock()
        
        state = {
            "messages": [
                HumanMessage(content="User question")
            ]
        }
        
        result = create_final_answer(state, mock_chat_model)
        
        assert "messages" in result
        assert "No tool results found" in result["messages"][0].content

    def test_create_final_answer_list_state(self):
        """Test final answer creation with list state."""
        mock_chat_model = Mock()
        
        state = [
            HumanMessage(content="User question"),
            ToolMessage(content="Tool result", tool_call_id="parsed-tool-call")
        ]
        
        with patch('legal_ai_assistant.agents.detect') as mock_detect, \
             patch('legal_ai_assistant.agents.calculate_max_response_tokens') as mock_calc:
            
            mock_detect.return_value = "en"
            mock_calc.return_value = 200
            mock_chat_model.invoke.return_value = Mock(content="Final answer")
            
            result = create_final_answer(state, mock_chat_model)
            
            assert "messages" in result
            assert result["messages"][0].content == "Final answer"

    @patch('legal_ai_assistant.agents.detect')
    def test_create_final_answer_error(self, mock_detect):
        """Test final answer creation with error."""
        mock_detect.side_effect = Exception("Detection error")
        
        mock_chat_model = Mock()
        mock_chat_model.invoke.side_effect = Exception("Model error")
        
        state = {
            "messages": [
                HumanMessage(content="User question"),
                ToolMessage(content="Tool result", tool_call_id="parsed-tool-call")
            ]
        }
        
        result = create_final_answer(state, mock_chat_model)
        
        assert "messages" in result
        assert "I apologize" in result["messages"][0].content


class TestRouteTools:
    """Test suite for route_tools function."""
    
    def test_route_tools_with_tool_calls(self):
        """Test routing when tool calls are present."""
        # Create a mock AI message with tool_calls attribute
        mock_ai_message = Mock()
        mock_ai_message.tool_calls = [{"name": "tool_rag", "args": {"query": "test"}}]
        
        state = {
            "messages": [mock_ai_message]
        }
        
        result = route_tools(state)
        assert result == "tools"

    def test_route_tools_with_tool_messages(self):
        """Test routing when tool messages are present."""
        state = {
            "messages": [
                HumanMessage(content="question"),
                ToolMessage(content="tool result", tool_call_id="test")
            ]
        }
        
        result = route_tools(state)
        assert result == "final_answer"

    def test_route_tools_no_tools(self):
        """Test routing when no tools are present."""
        state = {
            "messages": [
                HumanMessage(content="question")
            ]
        }
        
        result = route_tools(state)
        assert result == "__end__"

    def test_route_tools_list_state(self):
        """Test routing with list state."""
        state = [
            HumanMessage(content="question")
        ]
        
        result = route_tools(state)
        assert result == "__end__"

    def test_route_tools_empty_messages(self):
        """Test routing with empty messages."""
        with pytest.raises(ValueError):
            route_tools({"messages": []})


class TestUtilsAdvanced:
    """Advanced tests for utility functions."""
    
    @patch('legal_ai_assistant.utils.Chroma')
    def test_store_docs_with_embeddings(self, mock_chroma):
        """Test storing documents with embeddings."""
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        class MockDoc:
            def __init__(self, content):
                self.page_content = content
                self.metadata = {"source": "test.pdf"}
        
        docs = [MockDoc("test content")]
        
        result = store_docs_with_embeddings(docs)
        
        assert result == mock_vectorstore
        mock_chroma.from_documents.assert_called_once()

    @patch('legal_ai_assistant.utils.load_pdfs')
    @patch('legal_ai_assistant.utils.preprocess_pdfs')
    @patch('legal_ai_assistant.utils.load_or_create_vectorstore')
    def test_retrieve_documents(self, mock_load_vectorstore, mock_preprocess, mock_load_pdfs):
        """Test document retrieval."""
        # Mock documents
        mock_docs = [Mock()]
        mock_preprocess.return_value = mock_docs
        
        # Mock vectorstore and retriever
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = [
            Mock(page_content="relevant content", metadata={"source": "test.pdf"})
        ]
        mock_load_vectorstore.return_value = mock_vectorstore
        mock_vectorstore.as_retriever.return_value = mock_retriever
        
        result = retrieve_documents("test query", mock_retriever)
        
        assert len(result) > 0
        assert result[0].page_content == "relevant content"

    def test_retrieve_documents_empty_result(self):
        """Test document retrieval with empty result."""
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []
        
        result = retrieve_documents("test query", mock_retriever)
        
        assert len(result) == 0

    def test_retrieve_documents_with_limit(self):
        """Test document retrieval with content length limit."""
        # Create a real document-like object that can be modified
        class MockDoc:
            def __init__(self, content):
                self.page_content = content
                self.metadata = {"source": "test.pdf"}
        
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = [
            MockDoc("x" * 500)
        ]
        
        result = retrieve_documents("test query", mock_retriever)
        
        # Content should not be limited anymore (document truncation was removed)
        assert len(result[0].page_content) == 500  # Full content should be returned


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_token_calculation_edge_cases(self):
        """Test token calculation with edge cases."""
        from legal_ai_assistant.utils import calculate_max_response_tokens
        
        # Zero context (pass empty strings, not integers)
        result = calculate_max_response_tokens("", "")
        assert result > 0
        
        # Very large context (pass strings, not integers)
        result = calculate_max_response_tokens("Very large document content with many characters", "Very large user question")
        assert result > 0
        
    def test_parse_tool_call_edge_cases(self):
        """Test tool call parsing with edge cases."""
        from legal_ai_assistant.agents import parse_tool_call
        
        # Empty string
        result = parse_tool_call("")
        assert len(result) == 0
        
        # Malformed JSON
        result = parse_tool_call('{"name": "tool_rag"')  # Missing closing brace
        assert len(result) == 0
        
        # Valid JSON but wrong schema
        result = parse_tool_call('{"wrong": "schema"}')
        assert len(result) == 0

    def test_config_edge_cases(self):
        """Test configuration edge cases."""
        from legal_ai_assistant.config import LLM_CONFIG, EMBEDDING_CONFIG, APP_CONFIG
        
        # Ensure all configs are not empty
        assert len(LLM_CONFIG) > 0
        assert len(EMBEDDING_CONFIG) > 0
        assert len(APP_CONFIG) > 0
        
        # Ensure critical values are positive
        assert LLM_CONFIG["max_new_tokens"] > 0
        assert LLM_CONFIG["context_window"] > 0
