import requests
from bs4 import BeautifulSoup
import re
import time
from googleapiclient.discovery import build
from urllib.parse import urlparse
from nltk.tokenize import sent_tokenize


API_KEY = "AIzaSyBA-2V_K-sjI9voApCARrlKXZDdVgg5OPM"
SEARCH_ENGINE_ID = "c7df448be45934f6e"


class NewsArticleExtractor:
    def __init__(self):

        ####################### ba7out headers lma bb3t request #######################

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def extractUrl(self, url):
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            if any(domain in url for domain in ["bbc.com", "bbc.co.uk"]):
                # print("bbc hereeee")
                out = self.processBBC(response.text, url)
                return out
            elif any(domain in url for domain in ["arabnews.com"]):
                # print("arabnews hereeee")
                out = self.processArabsNews(response.text, url)
                return out
        except:
            return "error"
        
    def cleanText(self, text):
        text = re.sub(r'\s+', ' ', text)
        # basheel html comments
        text = re.sub(r'<!--.*?-->', '', text)
        # basheel script tags
        text = re.sub(r'<script.*?</script>', '', text)
        return text.strip()
    
    def processBBC(self, html_content, url):
        soup = BeautifulSoup(html_content, 'html.parser')
        
        result = {
            "source": "BBC News",
            "url": url,
            "content": "",
            "structured_content": None
        }
        
        article_text = []
        
        # ba7ded containers bto3y 3shan anadef the content
        contentContainers = soup.select('div[data-component="text-block"], div[data-component="subheadline-block"]')
        
        for container in contentContainers:
            header = container.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if header:
                article_text.append(f"## {header.get_text().strip()}")
                continue
                
            paragraphs = container.find_all('p')

            # hlf 3la kol paragraphs ely mwogdaa w ashouf ely msh fady

            for p in paragraphs:
                text = p.get_text().strip()
                if text:  
                    # ba7out text ely msh fadyy
                    article_text.append(text)
        
        result["content"] = "\n\n".join(article_text)
        
        result["structured_content"] = self.ExtractstructureContent(result["content"])
        
        return result
    
    def processArabsNews(self, html_content, url):

        # hstkhdm oparse bta3 soap w astkhdm parser bta3haa

        soup = BeautifulSoup(html_content, 'html.parser')
        
        result = {"source":"Arab News",
            "url":url,"content":"","structured_content":None
        }
        
        TITLE = soup.select_one('.title-area h2')
        if TITLE:
            result["title"] = TITLE.text.strip()
        
        date = soup.select_one('.entry-date time')
        if date:
            ##### basheel span tags w ageeb content 
            
            date_text = re.sub(r'<span>.*?</span>', '', str(date))
            date_text = BeautifulSoup(date_text, 'html.parser').text.strip()


            result["date"] = date_text

        articleText = []
        
        contentStruct = soup.select_one('.entry-content')
        if contentStruct:

            #####   extract header title w ba7to fi list 3ndy #####

            headers = contentStruct.find_all(['h1',
                                               'h2',
                                                 'h3', 
                                                 'h4', 
                                                 'h5', 'h6'])
            for header in headers:
                articleText.append(f"{header.get_text().strip()}")
                
            paragraphs = contentStruct.select('.field-item p')
            for p in paragraphs:
                if p.text.strip() and 'teads-adCall' not in p.get('class', []):
                    articleText.append(p.text.strip())
        
        result["content"] = "\n\n".join(articleText)
        
        result["structured_content"] = self.ExtractstructureContent(result["content"])
        
        return result
    def ExtractstructureContent(self, text_content):

######################### 3AYEZ ageeb structured content mn content #################

        paragraphs = text_content.split("\n\n")
        
        content = []
        section = {"title": "Introduction", "paragraphs": []}
        
        for paragraph in paragraphs:
            if paragraph.startswith("## "):
                if section["paragraphs"]:
                    content.append(section)
                
                section = {
                    "title": paragraph.replace("## ", "").strip(),
                    "paragraphs": []
                }
            else:
                clean_paragraph = re.sub(r'\s+', ' ', paragraph).strip()
                if clean_paragraph:
                    section["paragraphs"].append(clean_paragraph)
        
        if section["paragraphs"]:
            content.append(section)
            
        return content
    def getArticleText(self, content=None):
        if not content:
            return ""
            
        qa_text = []
        
        for section in content:
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
        
    
def search(query, numResults=10):
    
    ################### Ba3ml hena google search 3la websites bto3yyy #########################

    try:
        service = build("customsearch", "v1", developerKey=API_KEY)
        
        # Bazbt site resitrictions 3shan a3ml search query
        siterestrictions = " OR ".join([f"site:{domain}" for domain in ["bbc.com",  "arabnews.com"]])
       
        result = service.cse().list(q= f"{query} ({siterestrictions})", cx=SEARCH_ENGINE_ID, num=numResults).execute()
        
        if "items" not in result:
            return []
        
        domain_results = {domain: [] for domain in ["bbc.com",  "arabnews.com"]}
        
        for item in result["items"]:
            url = item["link"]
            for domain in ["bbc.com",  "arabnews.com"]:
                if domain in url:
                    domain_results[domain].append({"url": url, "title": item["title"]})
                    break
        
        return domain_results
    
    except:
        return {domain: [] for domain in ["bbc.com",  "arabnews.com"]}

def get_articles_for_query(query, maxDomain=2):
    results = search(query, maxDomain * 4)

    if not results:
        return []
    
    extractor = NewsArticleExtractor()
    articles = []
    
    mpDomain = {domain: 0 for domain in ["bbc.com",  "arabnews.com"]}
    
    for domain, results in results.items():
        for result in results:
            if mpDomain[domain] >= maxDomain:
                break
                
            time.sleep(1)
            
            article = extractor.extractUrl(result["url"])
            
            if article and "error" not in article and len(article["content"]) > 100:
                article["qa_text"] = extractor.getArticleText(article["structured_content"])
                articles.append(article)
                mpDomain[domain] += 1
    
    return articles