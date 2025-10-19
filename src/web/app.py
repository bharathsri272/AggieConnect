"""Streamlit web application for AggieConnect."""

import streamlit as st
import sys
from pathlib import Path
import time
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rag.pipeline import AggieConnectAssistant
from config import PROJECT_ROOT

# Page configuration
st.set_page_config(
    page_title="AggieConnect - UC Davis Student Assistant",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #002855 0%, #003d82 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        color: #e6f3ff;
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #002855;
    }
    
    .user-message {
        background-color: #e6f3ff;
        border-left-color: #002855;
    }
    
    .assistant-message {
        background-color: #f8f9fa;
        border-left-color: #28a745;
    }
    
    .help-topic {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #002855;
    }
    
    .stats-box {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'assistant' not in st.session_state:
    st.session_state.assistant = None
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_assistant():
    """Initialize the AggieConnect assistant."""
    if not st.session_state.initialized:
        with st.spinner("Initializing AggieConnect Assistant..."):
            try:
                st.session_state.assistant = AggieConnectAssistant()
                st.session_state.initialized = True
                st.success("✅ AggieConnect Assistant is ready!")
            except Exception as e:
                st.error(f"❌ Error initializing assistant: {str(e)}")
                st.session_state.assistant = None

def display_header():
    """Display the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🎓 AggieConnect</h1>
        <p>Your AI-powered UC Davis Student Assistant</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display the sidebar with help topics and stats."""
    with st.sidebar:
        st.markdown("## 📚 Help Topics")
        
        if st.session_state.assistant:
            help_topics = st.session_state.assistant.get_help_topics()
            for topic in help_topics:
                st.markdown(f"""
                <div class="help-topic">
                    {topic}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Conversation stats
        if st.session_state.conversation:
            st.markdown("## 📊 Conversation Stats")
            st.markdown(f"""
            <div class="stats-box">
                <strong>{len(st.session_state.conversation)}</strong><br>
                Questions Asked
            </div>
            """, unsafe_allow_html=True)
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation", type="secondary"):
            st.session_state.conversation = []
            st.rerun()
        
        st.markdown("---")
        
        # About section
        st.markdown("## ℹ️ About")
        st.markdown("""
        **AggieConnect** is an LLM-powered assistant designed to help UC Davis students access campus services, events, and academic resources.
        
        **Features:**
        - 🧠 Fine-tuned embedding models
        - 🔍 Semantic search with FAISS
        - 💬 Natural language responses
        - 📚 8,000+ campus FAQs and webpages
        """)

def display_conversation():
    """Display the conversation history."""
    if not st.session_state.conversation:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #666;">
            <h3>👋 Welcome to AggieConnect!</h3>
            <p>Ask me anything about UC Davis campus services, academics, or student life.</p>
            <p>Try asking: "How do I register for classes?" or "What dining options are available?"</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    for i, exchange in enumerate(st.session_state.conversation):
        # User message
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {exchange['question']}
        </div>
        """, unsafe_allow_html=True)
        
        # Assistant response
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>AggieConnect:</strong> {exchange['response']}
        </div>
        """, unsafe_allow_html=True)
        
        # Show sources if available
        if 'sources' in exchange and exchange['sources'] > 0:
            st.caption(f"📚 Based on {exchange['sources']} relevant sources")

def handle_user_input():
    """Handle user input and generate response."""
    user_input = st.chat_input("Ask me anything about UC Davis...")
    
    if user_input and st.session_state.assistant:
        # Add user message to conversation
        st.session_state.conversation.append({
            'question': user_input,
            'response': '',
            'sources': 0
        })
        
        # Generate response
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.assistant.ask(user_input)
                st.session_state.conversation[-1]['response'] = response
                
                # Get sources count from the last query
                if hasattr(st.session_state.assistant.rag_pipeline, 'last_query_result'):
                    result = st.session_state.assistant.rag_pipeline.last_query_result
                    st.session_state.conversation[-1]['sources'] = result.get('num_sources', 0)
                
            except Exception as e:
                st.session_state.conversation[-1]['response'] = f"Sorry, I encountered an error: {str(e)}"
        
        st.rerun()

def display_quick_questions():
    """Display quick question buttons."""
    st.markdown("### 💡 Quick Questions")
    
    quick_questions = [
        "How do I register for classes?",
        "What dining options are available?",
        "How can I apply for financial aid?",
        "What housing options are available?",
        "Where can I find study spaces?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(question, key=f"quick_{i}"):
                if st.session_state.assistant:
                    with st.spinner("Thinking..."):
                        response = st.session_state.assistant.ask(question)
                        st.session_state.conversation.append({
                            'question': question,
                            'response': response,
                            'sources': 0
                        })
                    st.rerun()

def main():
    """Main application function."""
    # Initialize assistant
    initialize_assistant()
    
    # Display header
    display_header()
    
    # Create layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Main chat area
        st.markdown("### 💬 Chat with AggieConnect")
        
        # Display conversation
        display_conversation()
        
        # Handle user input
        handle_user_input()
        
        # Quick questions
        if not st.session_state.conversation:
            display_quick_questions()
    
    with col2:
        # Sidebar
        display_sidebar()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🎓 AggieConnect - Powered by LLM and RAG Technology</p>
        <p>Built for UC Davis Students | Fine-tuned on 8,000+ Campus Resources</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
