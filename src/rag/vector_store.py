"""FAISS vector store implementation for AggieConnect."""

import faiss
import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from tqdm import tqdm

from config import FAISS_INDEX_PATH, EMBEDDING_DIMENSION, PROCESSED_DATA_DIR
from models.embedding_model import EmbeddingTrainer

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-based vector store for semantic search."""
    
    def __init__(self, embedding_model=None):
        self.index = None
        self.documents = []
        self.metadata = []
        self.embedding_model = embedding_model
        self.dimension = EMBEDDING_DIMENSION
        
        if self.embedding_model is None:
            self.embedding_model = EmbeddingTrainer()
            self.embedding_model.load_trained_model()
    
    def build_index(self, documents: List[Dict], faqs: List[Dict]) -> None:
        """Build FAISS index from documents and FAQs."""
        logger.info("Building FAISS index...")
        
        # Combine all text content
        all_texts = []
        all_metadata = []
        
        # Add FAQ questions and answers
        for faq in faqs:
            all_texts.append(faq['question'])
            all_metadata.append({
                'type': 'faq_question',
                'content': faq['question'],
                'answer': faq['answer'],
                'source_url': faq.get('source_url', ''),
                'page_title': faq.get('page_title', '')
            })
            
            all_texts.append(faq['answer'])
            all_metadata.append({
                'type': 'faq_answer',
                'content': faq['answer'],
                'question': faq['question'],
                'source_url': faq.get('source_url', ''),
                'page_title': faq.get('page_title', '')
            })
        
        # Add document content
        for doc in documents:
            all_texts.append(doc['content'])
            all_metadata.append({
                'type': 'document',
                'content': doc['content'],
                'title': doc['title'],
                'source_url': doc.get('source_url', ''),
                'chunk_index': doc.get('chunk_index', 0),
                'total_chunks': doc.get('total_chunks', 1)
            })
        
        logger.info(f"Computing embeddings for {len(all_texts)} texts...")
        
        # Compute embeddings
        embeddings = self.embedding_model.get_embeddings(all_texts)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents = all_texts
        self.metadata = all_metadata
        
        logger.info(f"Index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Get query embedding
        query_embedding = self.embedding_model.get_embeddings([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                results.append({
                    'content': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(score)
                })
        
        return results
    
    def save_index(self, index_path: str = None) -> None:
        """Save the FAISS index and metadata."""
        if index_path is None:
            index_path = FAISS_INDEX_PATH
        
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = index_path.parent / f"{index_path.stem}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        # Save documents
        docs_path = index_path.parent / f"{index_path.stem}_documents.json"
        with open(docs_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Index saved to {index_path}")
        logger.info(f"Metadata saved to {metadata_path}")
        logger.info(f"Documents saved to {docs_path}")
    
    def load_index(self, index_path: str = None) -> None:
        """Load the FAISS index and metadata."""
        if index_path is None:
            index_path = FAISS_INDEX_PATH
        
        index_path = Path(index_path)
        
        if not index_path.exists():
            logger.warning(f"Index file {index_path} not found")
            return
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = index_path.parent / f"{index_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        
        # Load documents
        docs_path = index_path.parent / f"{index_path.stem}_documents.json"
        if docs_path.exists():
            with open(docs_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
        
        logger.info(f"Index loaded with {self.index.ntotal} vectors")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if self.index is None:
            return {"status": "No index loaded"}
        
        stats = {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "total_documents": len(self.documents),
            "total_metadata": len(self.metadata)
        }
        
        # Count by type
        type_counts = {}
        for meta in self.metadata:
            doc_type = meta.get('type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        stats["type_counts"] = type_counts
        
        return stats


def build_vector_store() -> FAISSVectorStore:
    """Build and return a populated vector store."""
    # Load processed data
    faq_path = PROCESSED_DATA_DIR / "faqs.json"
    docs_path = PROCESSED_DATA_DIR / "documents.json"
    
    if not faq_path.exists() or not docs_path.exists():
        logger.warning("Processed data not found. Creating sample data...")
        # Create sample data
        sample_faqs = [
            {
                'question': 'How do I register for classes?',
                'answer': 'Use the Student Information System to register for classes.',
                'source_url': 'https://example.com',
                'type': 'faq',
                'page_title': 'Registration'
            }
        ]
        
        sample_docs = [
            {
                'id': 'doc1',
                'title': 'Student Services',
                'content': 'Comprehensive student services are available.',
                'source_url': 'https://example.com',
                'type': 'webpage',
                'chunk_index': 0,
                'total_chunks': 1
            }
        ]
        
        faqs, documents = sample_faqs, sample_docs
    else:
        with open(faq_path, 'r', encoding='utf-8') as f:
            faqs = json.load(f)
        
        with open(docs_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
    
    # Build vector store
    vector_store = FAISSVectorStore()
    vector_store.build_index(documents, faqs)
    
    # Save the index
    vector_store.save_index()
    
    return vector_store


def main():
    """Main function to build vector store."""
    vector_store = build_vector_store()
    
    # Test search
    test_queries = [
        "How do I register for classes?",
        "What dining options are available?",
        "Student services information"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vector_store.search(query, k=3)
        for i, result in enumerate(results):
            print(f"  {i+1}. Score: {result['score']:.3f}")
            print(f"     Type: {result['metadata']['type']}")
            print(f"     Content: {result['content'][:100]}...")
    
    # Print stats
    stats = vector_store.get_stats()
    print(f"\nVector Store Stats: {stats}")


if __name__ == "__main__":
    main()
