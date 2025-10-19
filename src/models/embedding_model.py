"""Embedding model training and fine-tuning for AggieConnect."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any
import logging
from tqdm import tqdm
import pickle

from config import (
    EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION, MAX_SEQUENCE_LENGTH,
    BATCH_SIZE, PROCESSED_DATA_DIR, EMBEDDINGS_DIR, CHECKPOINTS_DIR
)

logger = logging.getLogger(__name__)


class UCDavisDataset(Dataset):
    """Dataset for UC Davis FAQ and document pairs."""
    
    def __init__(self, faqs: List[Dict], documents: List[Dict], tokenizer, max_length: int = 512):
        self.faqs = faqs
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create training pairs
        self.pairs = self._create_training_pairs()
    
    def _create_training_pairs(self) -> List[Dict]:
        """Create positive and negative training pairs."""
        pairs = []
        
        # Create positive pairs from FAQs
        for faq in self.faqs:
            pairs.append({
                'text1': faq['question'],
                'text2': faq['answer'],
                'label': 1.0  # Positive pair
            })
        
        # Create positive pairs from document chunks (title + content)
        for doc in self.documents:
            if doc['type'] == 'webpage':
                pairs.append({
                    'text1': doc['title'],
                    'text2': doc['content'],
                    'label': 1.0  # Positive pair
                })
        
        # Create negative pairs (random combinations)
        import random
        random.seed(42)
        
        for _ in range(len(pairs) // 2):  # Half as many negative pairs
            faq1 = random.choice(self.faqs)
            faq2 = random.choice(self.faqs)
            if faq1 != faq2:
                pairs.append({
                    'text1': faq1['question'],
                    'text2': faq2['answer'],
                    'label': 0.0  # Negative pair
                })
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Tokenize both texts
        encoding1 = self.tokenizer(
            pair['text1'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoding2 = self.tokenizer(
            pair['text2'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids1': encoding1['input_ids'].squeeze(),
            'attention_mask1': encoding1['attention_mask'].squeeze(),
            'input_ids2': encoding2['input_ids'].squeeze(),
            'attention_mask2': encoding2['attention_mask'].squeeze(),
            'label': torch.tensor(pair['label'], dtype=torch.float)
        }


class EmbeddingModel(nn.Module):
    """Custom embedding model for UC Davis domain."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, embedding_dim: int = EMBEDDING_DIMENSION):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, embedding_dim)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        output = self.classifier(output)
        return output


class EmbeddingTrainer:
    """Trainer for fine-tuning embedding models."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load processed FAQ and document data."""
        faq_path = PROCESSED_DATA_DIR / "faqs.json"
        docs_path = PROCESSED_DATA_DIR / "documents.json"
        
        if not faq_path.exists() or not docs_path.exists():
            logger.warning("Processed data not found. Creating sample data...")
            return self._create_sample_data()
        
        with open(faq_path, 'r', encoding='utf-8') as f:
            faqs = json.load(f)
        
        with open(docs_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        logger.info(f"Loaded {len(faqs)} FAQs and {len(documents)} documents")
        return faqs, documents
    
    def _create_sample_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Create sample data for demonstration."""
        sample_faqs = [
            {
                'question': 'How do I register for classes?',
                'answer': 'Use the Student Information System to register for classes.',
                'source_url': 'https://example.com',
                'type': 'faq',
                'page_title': 'Registration'
            },
            {
                'question': 'What dining options are available?',
                'answer': 'Multiple dining commons and retail locations are available on campus.',
                'source_url': 'https://example.com',
                'type': 'faq',
                'page_title': 'Dining'
            }
        ]
        
        sample_docs = [
            {
                'id': 'doc1',
                'title': 'Student Services',
                'content': 'Comprehensive student services are available including academic advising and career counseling.',
                'source_url': 'https://example.com',
                'type': 'webpage',
                'chunk_index': 0,
                'total_chunks': 1
            }
        ]
        
        return sample_faqs, sample_docs
    
    def train_model(self, epochs: int = 3, learning_rate: float = 2e-5) -> None:
        """Train the embedding model."""
        logger.info("Starting model training...")
        
        # Load data
        faqs, documents = self.load_data()
        
        # Create dataset and dataloader
        dataset = UCDavisDataset(faqs, documents, self.tokenizer, MAX_SEQUENCE_LENGTH)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Initialize model
        self.model = EmbeddingModel(self.model_name).to(self.device)
        
        # Setup training
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                # Get embeddings for both texts
                emb1 = self.model(
                    batch['input_ids1'].to(self.device),
                    batch['attention_mask1'].to(self.device)
                )
                emb2 = self.model(
                    batch['input_ids2'].to(self.device),
                    batch['attention_mask2'].to(self.device)
                )
                
                # Compute cosine similarity
                cos_sim = torch.cosine_similarity(emb1, emb2, dim=1)
                
                # Compute loss
                loss = criterion(cos_sim, batch['label'].to(self.device))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Save model
        self.save_model()
        logger.info("Training completed!")
    
    def save_model(self) -> None:
        """Save the trained model."""
        if self.model is None:
            logger.error("No model to save")
            return
        
        model_path = EMBEDDINGS_DIR / "fine_tuned_embedding_model.pth"
        tokenizer_path = EMBEDDINGS_DIR / "tokenizer"
        
        # Save model state
        torch.save(self.model.state_dict(), model_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(tokenizer_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Tokenizer saved to {tokenizer_path}")
    
    def load_trained_model(self) -> None:
        """Load a previously trained model."""
        model_path = EMBEDDINGS_DIR / "fine_tuned_embedding_model.pth"
        tokenizer_path = EMBEDDINGS_DIR / "tokenizer"
        
        if not model_path.exists():
            logger.warning("No trained model found. Using base model.")
            self.model = EmbeddingModel(self.model_name).to(self.device)
            return
        
        # Load model
        self.model = EmbeddingModel(self.model_name).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        logger.info("Trained model loaded successfully!")
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts."""
        if self.model is None:
            self.load_trained_model()
        
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Computing embeddings"):
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=MAX_SEQUENCE_LENGTH,
                    return_tensors='pt'
                )
                
                embedding = self.model(
                    encoding['input_ids'].to(self.device),
                    encoding['attention_mask'].to(self.device)
                )
                
                embeddings.append(embedding.cpu().numpy())
        
        return np.vstack(embeddings)


class SentenceTransformerTrainer:
    """Alternative trainer using SentenceTransformers library."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def prepare_training_data(self, faqs: List[Dict], documents: List[Dict]) -> List[InputExample]:
        """Prepare training data for SentenceTransformers."""
        examples = []
        
        # Add FAQ pairs as positive examples
        for faq in faqs:
            examples.append(InputExample(
                texts=[faq['question'], faq['answer']],
                label=1.0
            ))
        
        # Add document title-content pairs
        for doc in documents:
            if doc['type'] == 'webpage':
                examples.append(InputExample(
                    texts=[doc['title'], doc['content']],
                    label=1.0
                ))
        
        return examples
    
    def train_sentence_transformer(self, epochs: int = 3) -> None:
        """Train using SentenceTransformers framework."""
        logger.info("Training SentenceTransformer model...")
        
        # Load data
        faqs, documents = self.load_data()
        train_examples = self.prepare_training_data(faqs, documents)
        
        # Create data loader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
        
        # Define loss function
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Train the model
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100,
            output_path=str(EMBEDDINGS_DIR / "sentence_transformer_model")
        )
        
        logger.info("SentenceTransformer training completed!")


def main():
    """Main function to run embedding model training."""
    # Option 1: Custom PyTorch training
    trainer = EmbeddingTrainer()
    trainer.train_model(epochs=2, learning_rate=2e-5)
    
    # Option 2: SentenceTransformers training (uncomment to use)
    # st_trainer = SentenceTransformerTrainer()
    # st_trainer.train_sentence_transformer(epochs=2)


if __name__ == "__main__":
    main()
