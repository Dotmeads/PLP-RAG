"""
Web scraper for fetching content from websites
"""
import requests
from bs4 import BeautifulSoup
import time
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebScraper:
    """Simple web scraper for extracting content from websites"""
    
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape content from a URL"""
        try:
            logger.info(f"Scraping URL: {url}")
            
            # Add delay to be respectful
            time.sleep(self.delay)
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title found"
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            # Extract metadata
            metadata = {
                'url': url,
                'title': title_text,
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'content_length': len(content)
            }
            
            return {
                'content': content,
                'metadata': metadata
            }
            
        except requests.RequestException as e:
            logger.error(f"Error fetching URL {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error parsing content from {url}: {str(e)}")
            return None
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from the page"""
        # Try to find main content areas
        main_selectors = [
            'main',
            'article',
            '.content',
            '.main-content',
            '.post-content',
            '.entry-content',
            '#content',
            '.container'
        ]
        
        main_content = None
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            # Fallback to body if no main content found
            main_content = soup.find('body')
        
        if main_content:
            # Extract text content
            text_content = main_content.get_text(separator='\n', strip=True)
            
            # Clean up the text
            lines = [line.strip() for line in text_content.split('\n')]
            lines = [line for line in lines if line and len(line) > 10]  # Remove very short lines
            
            return '\n\n'.join(lines)
        
        return "No content found"
    
    def save_content_to_file(self, content: str, filename: str, metadata: Dict[str, Any] = None):
        """Save scraped content to a file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                if metadata:
                    f.write(f"# {metadata.get('title', 'Scraped Content')}\n\n")
                    f.write(f"**Source:** {metadata.get('url', 'Unknown')}\n")
                    f.write(f"**Scraped:** {metadata.get('scraped_at', 'Unknown')}\n\n")
                    f.write("---\n\n")
                
                f.write(content)
            
            logger.info(f"Content saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving content to {filename}: {str(e)}")
            return False


if __name__ == "__main__":
    # Example usage
    scraper = WebScraper()
    
    # Scrape the SPCA pet care page
    url = "https://spca.org.sg/campaigns/pet-care-101/"
    result = scraper.scrape_url(url)
    
    if result:
        print(f"Title: {result['metadata']['title']}")
        print(f"Content length: {result['metadata']['content_length']} characters")
        print(f"Content preview: {result['content'][:500]}...")
        
        # Save to file
        scraper.save_content_to_file(
            result['content'], 
            "scraped_pet_care_content.txt", 
            result['metadata']
        )
    else:
        print("Failed to scrape content")
