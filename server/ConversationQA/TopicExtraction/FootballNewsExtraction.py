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

# Load environment variables
load_dotenv()

API_KEY = "AIzaSyBA-2V_K-sjI9voApCARrlKXZDdVgg5OPM"
SEARCH_ENGINE_ID = "c7df448be45934f6e"

# Only allow ESPN domain
ALLOWED_DOMAINS = ["espn.com"]

class NewsArticleExtractor:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def extract_from_url(self, url):
        if not self._is_allowed_domain(url):
            return {"error": f"URL not from allowed domains: {url}"}
            
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Process ESPN article
            return self._process_espn_article(response.text, url)
            
        except requests.RequestException as e:
            return {"error": f"Failed to fetch URL: {str(e)}"}
    
    def _is_allowed_domain(self, url):
        domain = urlparse(url).netloc
        return any(allowed_domain in domain for allowed_domain in ALLOWED_DOMAINS)
    
    def _clean_text(self, text):
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove HTML comment leftovers
        text = re.sub(r'<!--.*?-->', '', text)
        # Remove scripts leftovers
        text = re.sub(r'<script.*?</script>', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove any remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Fix spacing after periods, question marks, and exclamation points
        text = re.sub(r'([.!?])\s*', r'\1 ', text)
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        # Remove spaces before punctuation
        text = re.sub(r' ([,.;:])', r'\1', text)
        return text.strip()
    
    def _extract_text_without_links(self, element):
        if element is None:
            return ""
            
        text_parts = []
        for content in element.contents:
            if isinstance(content, str):
                text_parts.append(content)
            elif content.name == 'a':
                text_parts.append(content.get_text())
            elif content.name not in ['script', 'style', 'svg']:
                text_parts.append(self._extract_text_without_links(content))
                
        return ' '.join(text_parts)
    
    def _process_espn_article(self, html_content, url):
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for unwanted in soup.find_all(['script', 'style', 'noscript', 'svg', 
                                      'iframe', 'form', 'button', '.Ad', 
                                      '.Image__Citation', '.InlinePhoto']):
            unwanted.decompose()
        
        result = {
            "title": "",
            "date": "",
            "source": "ESPN FC",
            "url": url,
            "content": "",
            "structured_content": None,
            "football_data": {}
        }
        
        title_selectors = [
            'h1.article-header',
            '.headline',
            'h1.promo-title',
            '.article-info h1',
            '.GameInfo__Team',
            'h1',
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem and title_elem.text.strip():
                result["title"] = title_elem.text.strip()
                break
        
        date_selectors = [
            '.timestamp', 
            '.article-meta span', 
            '.date',
            'meta[name="date"]',
            'time'
        ]
        
        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                if selector == 'time' and date_elem.has_attr('datetime'):
                    result["date"] = date_elem['datetime']
                elif date_elem.text.strip():
                    result["date"] = date_elem.text.strip()
                break
        
        story_body = soup.find('div', class_=re.compile(r'Story__Body'))
        
        if story_body:
            article_text = self._extract_content_from_story_body(story_body)
        else:
            article_text = self._extract_content_from_other_containers(soup)
        
        # Extract football match data if available
        self._extract_football_data(soup, result)
        
        result["content"] = "\n\n".join(article_text) if article_text else ""
        
        if not result["content"]:
            result["content"] = self._extract_main_content(soup)
        
        result["structured_content"] = self._structure_content(result["content"])
        
        return result
    
    def _extract_content_from_story_body(self, story_body):
        article_text = []
        
        # Get all paragraphs
        paragraphs = story_body.find_all('p')
        for p in paragraphs:
            if not p.text.strip():
                continue
            
            p_text = self._extract_text_without_links(p)
            p_text = self._clean_text(p_text)
            
            if p_text:
                article_text.append(p_text)
        
        # Extract subheadings
        subheadings = story_body.find_all(['h2', 'h3', 'h4'])
        for heading in subheadings:
            heading_text = heading.get_text().strip()
            if heading_text:
                article_text.append(f"## {heading_text}")
        
        return article_text
    
    def _extract_content_from_other_containers(self, soup):
        article_text = []
        
        content_selectors = [
            '.article-body',
            '.story-content',
            '.article__body',
            '.game-details',
            'article',
            'main'
        ]
        
        content_elem = None
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                break
        
        if content_elem:
            # First remove any remaining unwanted elements
            for unwanted in content_elem.find_all(['.ad-container', '.inline-photo', 
                                                  'figcaption', '.image-caption', 
                                                  '.social-share', '.comments']):
                unwanted.decompose()
            
            # Extract paragraphs
            paragraphs = content_elem.find_all('p')
            for p in paragraphs:
                if not p.text.strip() or len(p.text.strip()) < 10:
                    continue
                
                # Extract clean text
                p_text = self._extract_text_without_links(p)
                p_text = self._clean_text(p_text)
                
                if p_text:
                    article_text.append(p_text)
            
            # Extract subheadings
            subheadings = content_elem.find_all(['h2', 'h3', 'h4'])
            for heading in subheadings:
                heading_text = heading.get_text().strip()
                if heading_text:
                    if article_text and not article_text[-1].startswith("##"):
                        article_text.append(f"## {heading_text}")
        
        return article_text
    
    def _extract_football_data(self, soup, result):
        """Extract football match data from the page"""
        # Look for match results/scores
        match_selectors = [
            '.competitors', 
            '.score-container', 
            '.game-details',
            '.Scoreboard',
            '.GameScoreBox'
        ]
        
        for selector in match_selectors:
            match_container = soup.select_one(selector)
            if match_container:
                # Try to extract team names
                teams = match_container.select('.team-name, .team, .ScoreCell__TeamName, .TeamName')
                # Try to extract scores
                scores = match_container.select('.score, .ScoreCell__Score, .Score')
                
                if teams and scores and len(teams) >= 2 and len(scores) >= 2:
                    result["football_data"]["match_result"] = {
                        "home_team": teams[0].text.strip() if teams else "",
                        "away_team": teams[1].text.strip() if len(teams) > 1 else "",
                        "home_score": scores[0].text.strip() if scores else "",
                        "away_score": scores[1].text.strip() if len(scores) > 1 else ""
                    }
                    break
        
        match_details = soup.select_one('.match-details, .game-stats, .GameInfo, .MatchStats')
        if match_details:
            stats_rows = match_details.select('.stats-row, .stat-row, .GameStats, .StatRow')
            stats_data = {}
            
            for row in stats_rows:
                label = row.select_one('.stats-label, .stat-label, .StatName')
                values = row.select('.stats-value, .stat-value, .StatValue')
                
                if label and values and len(values) >= 2:
                    stat_name = label.text.strip()
                    stats_data[stat_name] = {
                        "home": values[0].text.strip(),
                        "away": values[1].text.strip()
                    }
            
            if stats_data:
                result["football_data"]["match_stats"] = stats_data
    
    def _extract_main_content(self, soup):
        # Remove all unwanted elements first
        for unwanted in soup.find_all(['nav', 'header', 'footer', 'aside', 'iframe', 
                                    'script', 'style', 'noscript', 'form', 'figcaption',
                                    'div.Ad', '.Advertisement', '.ad-container', 
                                    '.Image__Caption', '.InlinePhoto']):
            unwanted.decompose()
        
        # Try to find content in common content containers
        content_tags = [
            soup.find('div', class_=re.compile(r'Story__Body')),
            soup.find('main'),
            soup.find('article'),
            soup.find('div', class_=re.compile(r'content|article|post|entry')),
            soup.find('div', id=re.compile(r'content|article|post|entry'))
        ]
        
        content = next((tag for tag in content_tags if tag is not None), soup.body)
        
        if content:
            paragraphs = []
            for p in content.find_all('p'):
                if p.text.strip() and len(p.text.strip()) > 10:
                    p_text = self._extract_text_without_links(p)
                    p_text = self._clean_text(p_text)
                    if p_text:
                        paragraphs.append(p_text)
            
            if paragraphs:
                return "\n\n".join(paragraphs)
            else:
                text = self._extract_text_without_links(content)
                return self._clean_text(text)
        
        return ""
    
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
                # Remove any extra whitespace or newlines
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
            qa_text.append(f"{section['title']}")
            
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
        
        site_restrictions = f"site:{ALLOWED_DOMAINS[0]}"
        football_focus = "football OR soccer"
        site_restricted_query = f"{query} {football_focus} {site_restrictions}"
        
        result = service.cse().list(q=site_restricted_query, cx=SEARCH_ENGINE_ID, num=num_results).execute()
        
        if "items" not in result:
            return []
        
        search_results = []
        for item in result["items"]:
            search_results.append({
                "url": item["link"],
                "title": item["title"],
                "snippet": item.get("snippet", "")
            })
        
        return search_results
    
    except Exception as e:
        print(f"Error performing Google search: {e}")
        return []
  
def get_football_articles(query, max_articles=3):
    search_results = google_search(query, max_articles * 2)

    if not search_results:
        return []
    
    extractor = NewsArticleExtractor()
    articles = []
    
    for i, result in enumerate(search_results):
        if len(articles) >= max_articles:
            break
            
        time.sleep(1)
        
        # Extract the article
        article = extractor.extract_from_url(result["url"])
        
        # Add the article if it has meaningful content
        if article and "error" not in article and len(article["content"]) > 100:
            # Get text formatted for QA
            article["qa_text"] = extractor.get_text_for_qa(article["structured_content"])
            articles.append(article)
    
    return articles