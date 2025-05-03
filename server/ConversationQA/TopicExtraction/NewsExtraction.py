import requests
from bs4 import BeautifulSoup
import re
import json
import time
from googleapiclient.discovery import build
from urllib.parse import urlparse
import os
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize

load_dotenv()

API_KEY = "AIzaSyBA-2V_K-sjI9voApCARrlKXZDdVgg5OPM"
SEARCH_ENGINE_ID = "c7df448be45934f6e"

ALLOWED_DOMAINS = ["bbc.com",  "arabnews.com"]

class NewsArticleExtractor:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def extract_from_url(self, url):
        """Extract article content from a URL"""
        if not self._is_allowed_domain(url):
            return {"error": f"URL not from allowed domains: {url}"}
            
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Determine the appropriate processor based on the domain
            if any(domain in url for domain in ["bbc.com", "bbc.co.uk"]):
                print("bbc hereeee")
                return self._process_bbc_article(response.text, url)
            elif any(domain in url for domain in ["arabnews.com"]):
                print("arabnews hereeee")
                return self._extract_arabnews_content(response.text, url)
            else:
                return self._process_generic_article(response.text, url)
        except requests.RequestException as e:
            return {"error": f"Failed to fetch URL: {str(e)}"}
    
    
    def _is_allowed_domain(self, url):
        return any(domain in url for domain in ALLOWED_DOMAINS)
    
    def _clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        # Remove HTML comment leftovers
        text = re.sub(r'<!--.*?-->', '', text)
        # Remove scripts leftovers
        text = re.sub(r'<script.*?</script>', '', text)
        return text.strip()
    
    def _process_bbc_article(self, html_content, url):
        soup = BeautifulSoup(html_content, 'html.parser')
        
        result = {
            "title": self._extract_title(soup),
            "date": self._extract_date(soup),
            "source": "BBC News",
            "url": url,
            "content": "",
            "structured_content": None
        }
        
        article_text = []
        
        content_containers = soup.select('div[data-component="text-block"], div[data-component="subheadline-block"]')
        
        for container in content_containers:
            header = container.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if header:
                article_text.append(f"## {header.get_text().strip()}")
                continue
                
            paragraphs = container.find_all('p')
            for p in paragraphs:
                text = p.get_text().strip()
                if text:  # Only add non-empty paragraphs
                    article_text.append(text)
        
        result["content"] = "\n\n".join(article_text)
        
        result["structured_content"] = self._structure_content(result["content"])
        
        return result
    
    def _extract_arabnews_content(self, html_content, url):
        soup = BeautifulSoup(html_content, 'html.parser')
        
        result = {
            "title": "",
            "date": "",
            "source": "Arab News",
            "url": url,
            "content": "",
            "structured_content": None
        }
        
        title_elem = soup.select_one('.title-area h2')
        if title_elem:
            result["title"] = title_elem.text.strip()
        
        date_elem = soup.select_one('.entry-date time')
        if date_elem:
            date_text = re.sub(r'<span>.*?</span>', '', str(date_elem))
            date_text = BeautifulSoup(date_text, 'html.parser').text.strip()
            result["date"] = date_text

        article_text = []
        
        content_elem = soup.select_one('.entry-content')
        if content_elem:
            headers = content_elem.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            for header in headers:
                article_text.append(f"## {header.get_text().strip()}")
                
            paragraphs = content_elem.select('.field-item p')
            for p in paragraphs:
                if p.text.strip() and 'teads-adCall' not in p.get('class', []):
                    article_text.append(p.text.strip())
        
        result["content"] = "\n\n".join(article_text)
        
        result["structured_content"] = self._structure_content(result["content"])
        
        return result

    def parse_arabnews_url(url):
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        url_info = {
            'domain': parsed_url.netloc,
            'section': path_parts[0] if len(path_parts) > 0 else '',
            'node_id': path_parts[1] if len(path_parts) > 1 else '',
        }
        
        return url_info

    def _process_generic_article(self, html_content, url):
        soup = BeautifulSoup(html_content, 'html.parser')
        
        result = {
            "title": self._extract_title(soup),
            "date": self._extract_date(soup),
            "source": self._extract_source(soup, url),
            "url": url,
            "content": "",
            "structured_content": None
        }
        
        content = self._extract_main_content(soup)
        result["content"] = content
        
        result["structured_content"] = self._structure_content(result["content"])
        
        return result
    
    def _extract_main_content(self, soup):
        content_tags = [
            soup.find('main'),
            soup.find('article'),
            soup.find('div', class_=re.compile(r'content|article|post|entry')),
            soup.find('div', id=re.compile(r'content|article|post|entry'))
        ]
        
        content = next((tag for tag in content_tags if tag is not None), soup.body)
        
        if content:
            for unwanted in content.find_all(['nav', 'header', 'footer', 'aside', 'iframe', 
                                           'script', 'style', 'noscript', 'form']):
                unwanted.decompose()
                
            text = content.get_text()
            return self._clean_text(text)
        
        return ""
    
    def _extract_title(self, soup):

        title_selectors = [
            'h1',
            'meta[property="og:title"]',
            'meta[name="twitter:title"]'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                if selector == 'h1':
                    return element.get_text().strip()
                else:
                    return element.get('content', '').strip()
        
        if soup.title:
            return soup.title.get_text().strip()
            
        return ""
    
    def _extract_date(self, soup):
        time_tag = soup.find('time')
        if time_tag and time_tag.has_attr('datetime'):
            return time_tag['datetime']
            
        date_selectors = [
            'meta[property="article:published_time"]',
            'meta[name="publish-date"]',
            'meta[name="date"]',
            'meta[name="DC.date.issued"]'
        ]
        
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element and element.has_attr('content'):
                return element['content']
                
        date_pattern = re.compile(r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}')
        text = soup.get_text()
        match = date_pattern.search(text)
        if match:
            return match.group(0)
            
        return ""
    
    def _extract_source(self, soup, url):
        source_selectors = [
            'meta[property="og:site_name"]',
            'meta[name="application-name"]'
        ]
        
        for selector in source_selectors:
            element = soup.select_one(selector)
            if element and element.has_attr('content'):
                return element['content']
        
        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if domain_match:
            domain = domain_match.group(1)
            if "bbc" in domain:
                return "BBC News"
            elif "reuters" in domain:
                return "Reuters"
            else:
                return domain.replace('.com', '').replace('.co.uk', '').capitalize()
                
        return "Unknown Source"
    
    def _structure_content(self, text_content):
        paragraphs = text_content.split("\n\n")
        
        structured_content = []
        current_section = {"title": "Introduction", "paragraphs": []}
        
        for paragraph in paragraphs:
            if paragraph.startswith("## "):
                if current_section["paragraphs"]:
                    structured_content.append(current_section)
                
                current_section = {
                    "title": paragraph.replace("## ", "").strip(),
                    "paragraphs": []
                }
            else:
                clean_paragraph = re.sub(r'\s+', ' ', paragraph).strip()
                if clean_paragraph:
                    current_section["paragraphs"].append(clean_paragraph)
        
        if current_section["paragraphs"]:
            structured_content.append(current_section)
            
        return structured_content
    
    def get_text_for_qa(self, structured_content=None):
        if not structured_content:
            return ""
            
        qa_text = []
        
        for section in structured_content:

            for paragraph in section['paragraphs']:
                qa_text.append(paragraph)
            
            qa_text.append("")
        
        return "\n".join(qa_text).strip()
    
    def split_into_sentences(self, text):
        try:
            sentences = sent_tokenize(text)
            return sentences
        except:
            return re.findall(r'[^.!?]+[.!?]', text)


def google_search(query, num_results=10):
    try:
        service = build("customsearch", "v1", developerKey=API_KEY)
        
        site_restrictions = " OR ".join([f"site:{domain}" for domain in ALLOWED_DOMAINS])
        site_restricted_query = f"{query} ({site_restrictions})"
        
        result = service.cse().list(q=site_restricted_query, cx=SEARCH_ENGINE_ID, num=num_results).execute()
        
        if "items" not in result:
            return []
        
        domain_results = {domain: [] for domain in ALLOWED_DOMAINS}
        
        for item in result["items"]:
            url = item["link"]
            for domain in ALLOWED_DOMAINS:
                if domain in url:
                    domain_results[domain].append({"url": url, "title": item["title"]})
                    break
        
        return domain_results
    
    except Exception as e:
        print(f"Error performing Google search: {e}")
        return {domain: [] for domain in ALLOWED_DOMAINS}
  
def get_articles_for_query(query, max_per_domain=2):
    search_results = google_search(query, max_per_domain * len(ALLOWED_DOMAINS) * 2)

    if not search_results:
        return []
    
    extractor = NewsArticleExtractor()
    articles = []
    
    domain_counts = {domain: 0 for domain in ALLOWED_DOMAINS}
    
    # Process each domain's results
    for domain, results in search_results.items():
        for result in results:
            if domain_counts[domain] >= max_per_domain:
                break
                
            time.sleep(1)
            
            
            article = extractor.extract_from_url(result["url"])
            
            if article and "error" not in article and len(article["content"]) > 100:
                article["qa_text"] = extractor.get_text_for_qa(article["structured_content"])
                articles.append(article)
                domain_counts[domain] += 1
    
    return articles