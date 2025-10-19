"""RAG pipeline implementation for AggieConnect."""

import openai
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from config import OPENAI_API_KEY
from rag.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for UC Davis assistant."""
    
    def __init__(self, vector_store: Optional[FAISSVectorStore] = None):
        self.vector_store = vector_store or FAISSVectorStore()
        self.openai_client = None
        
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
            self.openai_client = openai
        else:
            logger.warning("OpenAI API key not found. Using mock responses.")
    
    def retrieve_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context for the query."""
        try:
            results = self.vector_store.search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return []
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved results into context string."""
        if not results:
            return "No relevant information found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            content = result['content']
            score = result['score']
            
            if metadata['type'] == 'faq_question':
                context_parts.append(f"Q: {content}")
                if 'answer' in metadata:
                    context_parts.append(f"A: {metadata['answer']}")
            elif metadata['type'] == 'faq_answer':
                if 'question' in metadata:
                    context_parts.append(f"Q: {metadata['question']}")
                context_parts.append(f"A: {content}")
            else:
                context_parts.append(f"Information: {content}")
        
        return "\n\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using LLM with retrieved context."""
        if not self.openai_client:
            return self._generate_mock_response(query, context)
        
        try:
            prompt = self._create_prompt(query, context)
            
            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are AggieConnect, a helpful assistant for UC Davis students. Provide accurate, helpful information based on the context provided. If the context doesn't contain enough information, say so politely."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            return self._generate_mock_response(query, context)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for LLM."""
        return f"""Based on the following information about UC Davis, please answer the student's question.

Context:
{context}

Student's Question: {query}

Please provide a helpful, accurate response based on the context. If the context doesn't contain enough information to fully answer the question, please say so and provide what information you can."""
    
    def _generate_mock_response(self, query: str, context: str) -> str:
        """Generate mock response when OpenAI API is not available."""
        if "register" in query.lower() or "class" in query.lower():
            return "To register for classes at UC Davis, you can use the Student Information System (SIS). Log in with your UC Davis credentials and navigate to the registration section to add classes to your schedule."
        
        elif "dining" in query.lower() or "food" in query.lower():
            return "UC Davis offers multiple dining options including dining commons (Segundo, Tercero, Cuarto) and various retail dining locations. The dining commons provide all-you-care-to-eat meals with diverse options."
        
        elif "housing" in query.lower():
            return "UC Davis provides various housing options including residence halls for first-year students and apartment complexes for upperclassmen. Off-campus housing is also available in the Davis area."
        
        elif "financial aid" in query.lower():
            return "To apply for financial aid, complete the FAFSA or California Dream Act Application. Submit all required documents by the priority deadline and track your status through the Student Information System."
        
        else:
            return f"Based on the available information, I can help you with that. Here's what I found: {context[:200]}... For more specific information, please contact the relevant UC Davis office directly."
    
    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Main query method that combines retrieval and generation."""
        logger.info(f"Processing query: {question}")
        
        # Retrieve relevant context
        retrieved_results = self.retrieve_context(question, k=k)
        
        # Format context
        context = self.format_context(retrieved_results)
        
        # Generate response
        response = self.generate_response(question, context)
        
        return {
            'question': question,
            'response': response,
            'retrieved_context': retrieved_results,
            'context_string': context,
            'num_sources': len(retrieved_results)
        }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history (placeholder for future implementation)."""
        return []
    
    def add_to_conversation(self, question: str, response: str) -> None:
        """Add to conversation history (placeholder for future implementation)."""
        pass


class AggieConnectAssistant:
    """Main assistant class that wraps the RAG pipeline."""
    
    def __init__(self):
        self.rag_pipeline = RAGPipeline()
        self.conversation_history = []
    
    def ask(self, question: str) -> str:
        """Ask a question and get a response."""
        result = self.rag_pipeline.query(question)
        
        # Add to conversation history
        self.conversation_history.append({
            'question': question,
            'response': result['response'],
            'sources': result['num_sources']
        })
        
        return result['response']
    
    def get_help_topics(self) -> List[str]:
        """Get list of topics the assistant can help with."""
        return [
            "Class registration and enrollment",
            "Dining services and meal plans",
            "Housing and residence life",
            "Financial aid and scholarships",
            "Academic advising and support",
            "Campus services and resources",
            "Student life and activities",
            "Library services and study spaces",
            "Health and wellness services",
            "Career services and internships"
        ]
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history.copy()
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []


def main():
    """Main function to test the RAG pipeline."""
    assistant = AggieConnectAssistant()
    
    # Test queries
    test_questions = [
        "How do I register for classes?",
        "What dining options are available on campus?",
        "How can I apply for financial aid?",
        "What housing options are available for students?",
        "Where can I find study spaces on campus?"
    ]
    
    print("AggieConnect Assistant - UC Davis Student Helper")
    print("=" * 50)
    
    for question in test_questions:
        print(f"\nQ: {question}")
        response = assistant.ask(question)
        print(f"A: {response}")
        print("-" * 30)
    
    # Show conversation history
    print(f"\nConversation History ({len(assistant.get_conversation_history())} exchanges)")
    for i, exchange in enumerate(assistant.get_conversation_history(), 1):
        print(f"{i}. Q: {exchange['question']}")
        print(f"   A: {exchange['response'][:100]}...")
        print(f"   Sources: {exchange['sources']}")


if __name__ == "__main__":
    main()
