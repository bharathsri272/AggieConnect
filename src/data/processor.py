"""Data processing module for cleaning and preparing UC Davis data."""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
from tqdm import tqdm
import logging
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes and cleans collected UC Davis data."""
    
    def __init__(self):
        self.processed_documents = []
        self.faq_pairs = []
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Remove very short lines
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
        
        return ' '.join(lines).strip()
    
    def extract_faqs_from_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and clean FAQ pairs from collected data."""
        faqs = []
        
        for page in tqdm(data, desc="Extracting FAQs"):
            page_faqs = page.get('faqs', [])
            for faq in page_faqs:
                question = self.clean_text(faq.get('question', ''))
                answer = self.clean_text(faq.get('answer', ''))
                
                if len(question) > 10 and len(answer) > 20:
                    faqs.append({
                        'question': question,
                        'answer': answer,
                        'source_url': faq.get('source_url', ''),
                        'type': 'faq',
                        'page_title': page.get('title', '')
                    })
        
        return faqs
    
    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, 
                   overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def process_page_content(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process page content into searchable documents."""
        documents = []
        
        for page in tqdm(data, desc="Processing pages"):
            content = self.clean_text(page.get('content', ''))
            title = page.get('title', '')
            url = page.get('url', '')
            
            if len(content) < 50:  # Skip very short content
                continue
            
            # Create chunks from the content
            chunks = self.chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                documents.append({
                    'id': f"{url}_{i}",
                    'title': title,
                    'content': chunk,
                    'source_url': url,
                    'type': 'webpage',
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })
        
        return documents
    
    def create_sample_faqs(self) -> List[Dict[str, Any]]:
        """Create sample FAQ data for demonstration."""
        sample_faqs = [
            {
                'question': 'How do I register for classes at UC Davis?',
                'answer': 'You can register for classes through the Student Information System (SIS). Log in with your UC Davis credentials, navigate to the registration section, and follow the prompts to add classes to your schedule. Make sure to check prerequisites and course availability.',
                'source_url': 'https://www.ucdavis.edu/registration',
                'type': 'faq',
                'page_title': 'Registration Information'
            },
            {
                'question': 'What dining options are available on campus?',
                'answer': 'UC Davis offers multiple dining commons including Segundo, Tercero, and Cuarto, as well as various retail dining locations. The dining commons provide all-you-care-to-eat meals with diverse options including vegetarian, vegan, and allergen-free choices.',
                'source_url': 'https://www.ucdavis.edu/dining',
                'type': 'faq',
                'page_title': 'Dining Services'
            },
            {
                'question': 'How can I apply for financial aid?',
                'answer': 'To apply for financial aid, complete the Free Application for Federal Student Aid (FAFSA) or California Dream Act Application. Submit all required documents by the priority deadline. You can track your application status through the Student Information System.',
                'source_url': 'https://www.ucdavis.edu/financial-aid',
                'type': 'faq',
                'page_title': 'Financial Aid'
            },
            {
                'question': 'What housing options are available for students?',
                'answer': 'UC Davis offers residence halls, apartments, and family housing. First-year students typically live in residence halls, while upperclassmen can choose from various apartment complexes. Off-campus housing is also available in the surrounding Davis area.',
                'source_url': 'https://www.ucdavis.edu/housing',
                'type': 'faq',
                'page_title': 'Student Housing'
            },
            {
                'question': 'How do I access campus libraries and study spaces?',
                'answer': 'The UC Davis library system includes the Peter J. Shields Library and several specialized libraries. Students can access study spaces, computer labs, and research resources. Library cards are automatically linked to your student ID.',
                'source_url': 'https://www.ucdavis.edu/libraries',
                'type': 'faq',
                'page_title': 'Library Services'
            }
        ]
        
        return sample_faqs
    
    def process_data(self, input_file: str = "ucdavis_data.json") -> Tuple[List[Dict], List[Dict]]:
        """Main processing function."""
        input_path = RAW_DATA_DIR / input_file
        
        if not input_path.exists():
            logger.warning(f"Input file {input_path} not found. Creating sample data.")
            # Create sample data for demonstration
            sample_faqs = self.create_sample_faqs()
            sample_documents = [
                {
                    'id': 'sample_1',
                    'title': 'UC Davis Student Services',
                    'content': 'UC Davis provides comprehensive student services including academic advising, career counseling, health services, and mental health support. Students can access these services through various campus offices and online portals.',
                    'source_url': 'https://www.ucdavis.edu/student-services',
                    'type': 'webpage',
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            ]
            
            self.faq_pairs = sample_faqs
            self.processed_documents = sample_documents
            
        else:
            # Load and process real data
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Processing {len(data)} pages...")
            
            # Extract FAQs
            self.faq_pairs = self.extract_faqs_from_data(data)
            
            # Process page content
            self.processed_documents = self.process_page_content(data)
        
        logger.info(f"Processed {len(self.faq_pairs)} FAQs and {len(self.processed_documents)} documents")
        
        return self.faq_pairs, self.processed_documents
    
    def save_processed_data(self) -> None:
        """Save processed data to files."""
        # Save FAQs
        faq_path = PROCESSED_DATA_DIR / "faqs.json"
        with open(faq_path, 'w', encoding='utf-8') as f:
            json.dump(self.faq_pairs, f, indent=2, ensure_ascii=False)
        
        # Save documents
        docs_path = PROCESSED_DATA_DIR / "documents.json"
        with open(docs_path, 'w', encoding='utf-8') as f:
            json.dump(self.processed_documents, f, indent=2, ensure_ascii=False)
        
        # Save as CSV for easy viewing
        faq_df = pd.DataFrame(self.faq_pairs)
        faq_csv_path = PROCESSED_DATA_DIR / "faqs.csv"
        faq_df.to_csv(faq_csv_path, index=False)
        
        docs_df = pd.DataFrame(self.processed_documents)
        docs_csv_path = PROCESSED_DATA_DIR / "documents.csv"
        docs_df.to_csv(docs_csv_path, index=False)
        
        logger.info(f"Processed data saved to {PROCESSED_DATA_DIR}")
        logger.info(f"  - FAQs: {len(self.faq_pairs)}")
        logger.info(f"  - Documents: {len(self.processed_documents)}")


def main():
    """Main function to run data processing."""
    processor = DataProcessor()
    processor.process_data()
    processor.save_processed_data()


if __name__ == "__main__":
    main()
