from newsapi import NewsApiClient
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def fetch_news(ticker, company_name):
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)  # Get free key from newsapi.org
    articles = newsapi.get_everything(q=company_name, language='en', sort_by='publishedAt', page_size=5)
    
    docs = []
    for article in articles['articles']:
        content = f"{article['title']}. {article['description']}"
        docs.append(Document(page_content=content, metadata={"source": article['url'], "date": article['publishedAt']}))
    return docs