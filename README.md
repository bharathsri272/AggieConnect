# AggieConnect

An LLM-powered assistant to help UC Davis students access campus services, events, and academic resources using a retrieval-augmented generation (RAG) pipeline.

## Features

- **Intelligent Campus Assistant**: Powered by fine-tuned embedding models and LLMs
- **Comprehensive Knowledge Base**: Processes 8,000+ campus FAQs and webpages
- **Fast Semantic Search**: FAISS-powered retrieval for relevant information
- **Student-Focused**: Tailored responses for UC Davis campus services and resources

## Architecture

- **Data Collection**: Automated scraping of campus FAQs and webpages
- **Embedding Models**: Fine-tuned PyTorch models for domain-specific embeddings
- **RAG Pipeline**: Retrieval-augmented generation with FAISS vector search
- **LLM Integration**: OpenAI GPT models for natural language responses
- **Web Interface**: Streamlit-based chat interface

## Project Structure

```
AggieConnect/
├── src/
│   ├── data/           # Data collection and processing
│   ├── models/         # Embedding model training and inference
│   ├── rag/           # RAG pipeline implementation
│   ├── api/           # FastAPI backend
│   └── web/           # Streamlit frontend
├── data/
│   ├── raw/           # Raw scraped data
│   └── processed/     # Processed and cleaned data
├── models/
│   ├── embeddings/    # Trained embedding models
│   └── checkpoints/   # Model checkpoints
├── tests/             # Unit and integration tests
├── docs/              # Documentation
└── notebooks/         # Jupyter notebooks for experimentation
```

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables (see `.env.example`)
4. Run data collection: `python src/data/collector.py`
5. Train embedding models: `python src/models/train_embeddings.py`
6. Start the web interface: `streamlit run src/web/app.py`

## Usage

The system provides a conversational interface where students can ask questions about:
- Campus services and resources
- Academic policies and procedures
- Events and activities
- Housing and dining information
- Financial aid and registration

## Team

Developed by a 4-person team demonstrating how machine learning and LLMs can enhance student access to information at UC Davis.
