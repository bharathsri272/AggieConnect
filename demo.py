"""Demo script for AggieConnect."""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from rag.pipeline import AggieConnectAssistant
from data.processor import DataProcessor
from rag.vector_store import build_vector_store


def run_demo():
    """Run a comprehensive demo of AggieConnect."""
    print("🎓 AggieConnect Demo - UC Davis Student Assistant")
    print("=" * 60)
    
    # Step 1: Data Processing Demo
    print("\n📊 Step 1: Processing Campus Data...")
    processor = DataProcessor()
    faqs, documents = processor.process_data()
    print(f"✅ Processed {len(faqs)} FAQs and {len(documents)} documents")
    
    # Step 2: Vector Store Demo
    print("\n🔍 Step 2: Building Vector Store...")
    vector_store = build_vector_store()
    stats = vector_store.get_stats()
    print(f"✅ Vector store built with {stats['total_vectors']} vectors")
    
    # Step 3: Assistant Demo
    print("\n🤖 Step 3: Initializing Assistant...")
    assistant = AggieConnectAssistant()
    print("✅ Assistant initialized successfully!")
    
    # Step 4: Interactive Demo
    print("\n💬 Step 4: Interactive Demo")
    print("Ask questions about UC Davis campus services, academics, or student life.")
    print("Type 'quit' to exit, 'help' for topics, 'history' for conversation history.")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n🎓 You: ").strip()
            
            if question.lower() == 'quit':
                print("👋 Goodbye! Thanks for trying AggieConnect!")
                break
            
            elif question.lower() == 'help':
                print("\n📚 Help Topics:")
                for i, topic in enumerate(assistant.get_help_topics(), 1):
                    print(f"  {i}. {topic}")
                continue
            
            elif question.lower() == 'history':
                history = assistant.get_conversation_history()
                if history:
                    print(f"\n📜 Conversation History ({len(history)} exchanges):")
                    for i, exchange in enumerate(history, 1):
                        print(f"  {i}. Q: {exchange['question']}")
                        print(f"     A: {exchange['response'][:100]}...")
                else:
                    print("\n📜 No conversation history yet.")
                continue
            
            elif not question:
                continue
            
            # Get response from assistant
            print("🤔 Thinking...")
            response = assistant.ask(question)
            print(f"🎓 AggieConnect: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Final stats
    print(f"\n📊 Demo Summary:")
    print(f"  - Questions asked: {len(assistant.get_conversation_history())}")
    print(f"  - Vector store size: {stats['total_vectors']} vectors")
    print(f"  - Document types: {list(stats['type_counts'].keys())}")


def run_quick_demo():
    """Run a quick demo with predefined questions."""
    print("�� AggieConnect Quick Demo")
    print("=" * 40)
    
    assistant = AggieConnectAssistant()
    
    demo_questions = [
        "How do I register for classes at UC Davis?",
        "What dining options are available on campus?",
        "How can I apply for financial aid?",
        "What housing options are available for students?",
        "Where can I find study spaces on campus?"
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n{i}. Q: {question}")
        response = assistant.ask(question)
        print(f"   A: {response}")
        print("-" * 40)
    
    print(f"\n✅ Demo completed! Asked {len(demo_questions)} questions.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AggieConnect Demo")
    parser.add_argument("--quick", action="store_true", help="Run quick demo with predefined questions")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_demo()
    else:
        run_demo()
