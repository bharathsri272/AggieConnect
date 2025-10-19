"""Tests for the RAG pipeline."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.pipeline import RAGPipeline, AggieConnectAssistant
from rag.vector_store import FAISSVectorStore


class TestRAGPipeline:
    """Test cases for RAG pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = RAGPipeline()
        assert pipeline is not None
        assert pipeline.vector_store is not None
    
    def test_mock_response_generation(self):
        """Test mock response generation when OpenAI is not available."""
        pipeline = RAGPipeline()
        
        # Test registration query
        response = pipeline._generate_mock_response(
            "How do I register for classes?",
            "Registration information"
        )
        assert "register" in response.lower() or "sis" in response.lower()
        
        # Test dining query
        response = pipeline._generate_mock_response(
            "What dining options are available?",
            "Dining information"
        )
        assert "dining" in response.lower()
    
    def test_context_formatting(self):
        """Test context formatting."""
        pipeline = RAGPipeline()
        
        mock_results = [
            {
                'content': 'Test question',
                'metadata': {
                    'type': 'faq_question',
                    'answer': 'Test answer'
                },
                'score': 0.9
            }
        ]
        
        context = pipeline.format_context(mock_results)
        assert "Test question" in context
        assert "Test answer" in context


class TestAggieConnectAssistant:
    """Test cases for AggieConnect assistant."""
    
    def test_assistant_initialization(self):
        """Test assistant initialization."""
        assistant = AggieConnectAssistant()
        assert assistant is not None
        assert assistant.rag_pipeline is not None
        assert assistant.conversation_history == []
    
    def test_help_topics(self):
        """Test help topics retrieval."""
        assistant = AggieConnectAssistant()
        topics = assistant.get_help_topics()
        
        assert isinstance(topics, list)
        assert len(topics) > 0
        assert any("registration" in topic.lower() for topic in topics)
        assert any("dining" in topic.lower() for topic in topics)
    
    def test_conversation_history(self):
        """Test conversation history management."""
        assistant = AggieConnectAssistant()
        
        # Initially empty
        assert len(assistant.get_conversation_history()) == 0
        
        # Add a conversation
        assistant.conversation_history.append({
            'question': 'Test question',
            'response': 'Test response',
            'sources': 1
        })
        
        history = assistant.get_conversation_history()
        assert len(history) == 1
        assert history[0]['question'] == 'Test question'
        
        # Clear history
        assistant.clear_history()
        assert len(assistant.get_conversation_history()) == 0


class TestFAISSVectorStore:
    """Test cases for FAISS vector store."""
    
    def test_vector_store_initialization(self):
        """Test vector store initialization."""
        vector_store = FAISSVectorStore()
        assert vector_store is not None
        assert vector_store.embedding_model is not None
    
    def test_stats_without_index(self):
        """Test stats when no index is loaded."""
        vector_store = FAISSVectorStore()
        stats = vector_store.get_stats()
        assert stats["status"] == "No index loaded"


if __name__ == "__main__":
    pytest.main([__file__])
