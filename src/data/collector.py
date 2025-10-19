"""Data collection module for UC Davis campus information."""

import requests
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse
import logging
from tqdm import tqdm

from config import RAW_DATA_DIR, UC_DAVIS_FAQ_URL, UC_DAVIS_SERVICES_URL, UC_DAVIS_ACADEMICS_URL

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UC DavisDataCollector:
    """Collects data from UC Davis websites and FAQs."""
    
    def __init__(self, base_urls: List[str], max_pages: int = 100):
        self.base_urls = base_urls
        self.max_pages = max_pages
        self.visited_urls = set()
        self.collected_data = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and from UC Davis domain."""
        try:
            parsed = urlparse(url)
            return parsed.netloc.endswith('ucdavis.edu') and url not in self.visited_urls
        except:
            return False
    
    def extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from HTML."""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_faqs(self, soup: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """Extract FAQ pairs from the page."""
        faqs = []
        
        # Look for common FAQ patterns
        faq_selectors = [
            'div.faq-item',
            'div.faq',
            'div.question-answer',
            'div.accordion-item',
            'div[class*="faq"]',
            'div[class*="question"]'
        ]
        
        for selector in faq_selectors:
            elements = soup.select(selector)
            for element in elements:
                question_elem = element.find(['h3', 'h4', 'h5', 'div.question', 'span.question'])
                answer_elem = element.find(['div.answer', 'div.content', 'p'])
                
                if question_elem and answer_elem:
                    question = question_elem.get_text().strip()
                    answer = answer_elem.get_text().strip()
                    
                    if len(question) > 10 and len(answer) > 20:
                        faqs.append({
                            'question': question,
                            'answer': answer,
                            'source_url': url,
                            'type': 'faq'
                        })
        
        return faqs
    
    def scrape_page(self, url: str) -> Dict[str, Any]:
        """Scrape a single page and extract relevant information."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract basic page information
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title"
            
            # Extract main content
            main_content = soup.find('main') or soup.find('div', class_='content') or soup.find('body')
            content = self.extract_text_content(main_content) if main_content else ""
            
            # Extract FAQs if present
            faqs = self.extract_faqs(soup, url)
            
            page_data = {
                'url': url,
                'title': title_text,
                'content': content,
                'faqs': faqs,
                'word_count': len(content.split()),
                'timestamp': time.time()
            }
            
            return page_data
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None
    
    def find_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Find relevant links on the page."""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            if self.is_valid_url(full_url):
                # Filter for relevant pages
                if any(keyword in full_url.lower() for keyword in 
                      ['faq', 'help', 'service', 'academic', 'student', 'campus']):
                    links.append(full_url)
        
        return links
    
    def collect_from_url(self, url: str, depth: int = 0, max_depth: int = 2) -> None:
        """Recursively collect data from a URL and its links."""
        if depth > max_depth or len(self.visited_urls) >= self.max_pages:
            return
        
        if url in self.visited_urls:
            return
        
        self.visited_urls.add(url)
        logger.info(f"Scraping: {url} (depth: {depth})")
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Scrape current page
            page_data = self.scrape_page(url)
            if page_data:
                self.collected_data.append(page_data)
            
            # Find and follow links
            if depth < max_depth:
                links = self.find_links(soup, url)
                for link in links[:10]:  # Limit links per page
                    time.sleep(1)  # Be respectful
                    self.collect_from_url(link, depth + 1, max_depth)
            
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
        
        time.sleep(1)  # Rate limiting
    
    def collect_all_data(self) -> List[Dict[str, Any]]:
        """Collect data from all base URLs."""
        logger.info("Starting data collection from UC Davis websites...")
        
        for base_url in self.base_urls:
            logger.info(f"Collecting from: {base_url}")
            self.collect_from_url(base_url)
            time.sleep(2)  # Rate limiting between base URLs
        
        logger.info(f"Collection complete. Found {len(self.collected_data)} pages.")
        return self.collected_data
    
    def save_data(self, filename: str = "ucdavis_data.json") -> None:
        """Save collected data to JSON file."""
        output_path = RAW_DATA_DIR / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.collected_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Data saved to: {output_path}")
        
        # Print summary
        total_faqs = sum(len(page.get('faqs', [])) for page in self.collected_data)
        total_words = sum(page.get('word_count', 0) for page in self.collected_data)
        
        logger.info(f"Summary:")
        logger.info(f"  - Pages collected: {len(self.collected_data)}")
        logger.info(f"  - FAQs found: {total_faqs}")
        logger.info(f"  - Total words: {total_words}")


def main():
    """Main function to run data collection."""
    base_urls = [
        UC_DAVIS_FAQ_URL,
        UC_DAVIS_SERVICES_URL,
        UC_DAVIS_ACADEMICS_URL,
        "https://www.ucdavis.edu/student-life",
        "https://www.ucdavis.edu/admissions",
        "https://www.ucdavis.edu/financial-aid",
        "https://www.ucdavis.edu/housing",
        "https://www.ucdavis.edu/dining"
    ]
    
    collector = UC DavisDataCollector(base_urls, max_pages=200)
    collector.collect_all_data()
    collector.save_data()


if __name__ == "__main__":
    main()
