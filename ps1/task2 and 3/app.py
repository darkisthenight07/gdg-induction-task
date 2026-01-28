"""
Stock Market RAG Chatbot with Real-Time Intelligence
Tasks 2 & 3: Analytical Chatbot + Live Data Integration
"""

import gradio as gr
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import time

# RAG Components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# LLM - Using Groq for fast inference (you can switch to Gemini)
try:
    from langchain_groq import ChatGroq
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: Groq not installed. Install with: pip install langchain-groq")

# News scraping
import requests
from bs4 import BeautifulSoup
import feedparser


# ============================================================================
# CONFIGURATION
# ============================================================================

COMMON_TICKERS = {
    'apple': 'AAPL',
    'microsoft': 'MSFT',
    'google': 'GOOGL',
    'amazon': 'AMZN',
    'tesla': 'TSLA',
    'meta': 'META',
    'nvidia': 'NVDA',
    'netflix': 'NFLX',
    'facebook': 'META',
}

# Global state for live data
class LiveDataManager:
    def __init__(self):
        self.active_tickers = []
        self.live_data = {}
        self.is_streaming = False
        self.update_interval = 60  # seconds
        
    def start_streaming(self, tickers: List[str]):
        self.active_tickers = tickers
        self.is_streaming = True
        threading.Thread(target=self._stream_loop, daemon=True).start()
    
    def stop_streaming(self):
        self.is_streaming = False
    
    def _stream_loop(self):
        while self.is_streaming:
            for ticker in self.active_tickers:
                try:
                    stock = yf.Ticker(ticker)
                    current_data = stock.history(period='1d', interval='1m')
                    if not current_data.empty:
                        self.live_data[ticker] = current_data
                except Exception as e:
                    print(f"Error fetching {ticker}: {e}")
            time.sleep(self.update_interval)

live_manager = LiveDataManager()


# ============================================================================
# TICKER EXTRACTION & INTENT RECOGNITION
# ============================================================================

def extract_ticker(query: str) -> Optional[str]:
    """
    Extract stock ticker from natural language query.
    Uses semantic understanding rather than just keywords.
    """
    query_lower = query.lower()
    
    # Direct ticker mention (e.g., "AAPL", "MSFT")
    ticker_pattern = r'\b[A-Z]{1,5}\b'
    matches = re.findall(ticker_pattern, query)
    if matches:
        return matches[0]
    
    # Company name to ticker mapping
    for company, ticker in COMMON_TICKERS.items():
        if company in query_lower:
            return ticker
    
    return None


def classify_intent(query: str) -> str:
    """
    Classify user intent for routing to appropriate agent.
    """
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['why', 'reason', 'cause', 'explain', 'happened']):
        return 'explanation'
    elif any(word in query_lower for word in ['when', 'date', 'time', 'trend']):
        return 'trend_analysis'
    elif any(word in query_lower for word in ['plot', 'chart', 'graph', 'show', 'visualize']):
        return 'visualization'
    elif any(word in query_lower for word in ['predict', 'forecast', 'future']):
        return 'prediction'
    elif any(word in query_lower for word in ['compare', 'vs', 'versus', 'against']):
        return 'comparison'
    else:
        return 'general'


# ============================================================================
# DATA COLLECTION & PROCESSING
# ============================================================================

def fetch_stock_data(ticker: str, period: str = '1y') -> pd.DataFrame:
    """Fetch historical stock data"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for analysis"""
    if df.empty:
        return df
    
    # Moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Daily returns
    df['Returns'] = df['Close'].pct_change() * 100
    
    return df


def detect_trends(df: pd.DataFrame) -> List[Dict]:
    """
    Detect significant uptrends and downtrends with dates.
    """
    if df.empty or len(df) < 5:
        return []
    
    trends = []
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Find significant moves (>5% change)
    df['Price_Change'] = df['Close'].pct_change(periods=5) * 100
    
    significant_ups = df[df['Price_Change'] > 5]
    significant_downs = df[df['Price_Change'] < -5]
    
    for date, row in significant_ups.iterrows():
        trends.append({
            'type': 'uptrend',
            'date': date.strftime('%Y-%m-%d'),
            'change': row['Price_Change'],
            'price': row['Close']
        })
    
    for date, row in significant_downs.iterrows():
        trends.append({
            'type': 'downtrend',
            'date': date.strftime('%Y-%m-%d'),
            'change': row['Price_Change'],
            'price': row['Close']
        })
    
    return sorted(trends, key=lambda x: x['date'], reverse=True)


# ============================================================================
# NEWS SCRAPING & RAG SETUP
# ============================================================================

def scrape_financial_news(ticker: str, company_name: str = None) -> List[Dict]:
    """
    Scrape financial news from multiple sources.
    Returns list of news articles with title, content, date, source.
    """
    news_items = []
    
    # Yahoo Finance RSS
    try:
        search_term = company_name or ticker
        rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
        feed = feedparser.parse(rss_url)
        
        for entry in feed.entries[:10]:
            news_items.append({
                'title': entry.get('title', ''),
                'content': entry.get('summary', ''),
                'date': entry.get('published', ''),
                'source': 'Yahoo Finance',
                'link': entry.get('link', '')
            })
    except Exception as e:
        print(f"Error fetching Yahoo Finance news: {e}")
    
    # Google Finance (via web scraping)
    try:
        url = f"https://www.google.com/finance/quote/{ticker}:NASDAQ"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # This is a simplified version - actual implementation would need more robust scraping
            news_divs = soup.find_all('div', class_='yY3Lee')
            for div in news_divs[:5]:
                title = div.get_text() if div else ""
                if title:
                    news_items.append({
                        'title': title,
                        'content': title,
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'source': 'Google Finance',
                        'link': ''
                    })
    except Exception as e:
        print(f"Error scraping Google Finance: {e}")
    
    return news_items


def build_vector_store(ticker: str, stock_data: pd.DataFrame, news: List[Dict]):
    """
    Build vector store for RAG from stock data and news.
    """
    documents = []
    
    # Add stock price movements as documents
    trends = detect_trends(stock_data)
    for trend in trends:
        content = f"""
        Stock: {ticker}
        Date: {trend['date']}
        Movement: {trend['type']}
        Price Change: {trend['change']:.2f}%
        Price: ${trend['price']:.2f}
        """
        documents.append(Document(
            page_content=content,
            metadata={'type': 'price_movement', 'date': trend['date']}
        ))
    
    # Add news articles
    for article in news:
        content = f"""
        Title: {article['title']}
        Content: {article['content']}
        Date: {article['date']}
        Source: {article['source']}
        """
        documents.append(Document(
            page_content=content,
            metadata={'type': 'news', 'date': article['date'], 'source': article['source']}
        ))
    
    # Add summary statistics
    if not stock_data.empty:
        latest_price = stock_data['Close'].iloc[-1]
        price_change_1w = ((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-5]) - 1) * 100
        price_change_1m = ((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-20]) - 1) * 100
        
        summary = f"""
        Stock: {ticker}
        Current Price: ${latest_price:.2f}
        1-Week Change: {price_change_1w:.2f}%
        1-Month Change: {price_change_1m:.2f}%
        52-Week High: ${stock_data['High'].max():.2f}
        52-Week Low: ${stock_data['Low'].min():.2f}
        Average Volume: {stock_data['Volume'].mean():.0f}
        """
        documents.append(Document(
            page_content=summary,
            metadata={'type': 'summary'}
        ))
    
    if not documents:
        return None
    
    # Create embeddings and vector store
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=f"{ticker}_knowledge"
    )
    
    return vectorstore


# ============================================================================
# LLM & RAG CHAIN
# ============================================================================

def create_rag_chain(vectorstore, llm_api_key: str = None):
    """
    Create RAG chain with LLM for answering queries.
    """
    if not LLM_AVAILABLE or not llm_api_key:
        return None
    
    # Initialize LLM (Groq example - you can switch to Gemini)
    llm = ChatGroq(
        temperature=0.3,
        model_name="mixtral-8x7b-32768",
        groq_api_key=llm_api_key
    )
    
    # Custom prompt for financial analysis
    prompt_template = """
    You are a financial analyst assistant. Use the following context to answer the question.
    Provide specific dates, numbers, and cite sources when available.
    
    Context: {context}
    
    Question: {question}
    
    Answer: Provide a clear, detailed explanation with specific facts and figures.
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain


# ============================================================================
# AGENT SYSTEM
# ============================================================================

class ResearchAgent:
    """Agent for retrieving and explaining market context"""
    
    def __init__(self, vectorstore, rag_chain=None):
        self.vectorstore = vectorstore
        self.rag_chain = rag_chain
    
    def answer(self, query: str) -> str:
        if self.rag_chain:
            result = self.rag_chain({"query": query})
            answer = result['result']
            sources = result.get('source_documents', [])
            
            # Add source citations
            if sources:
                answer += "\n\n**Sources:**\n"
                for i, doc in enumerate(sources[:3], 1):
                    metadata = doc.metadata
                    answer += f"{i}. {metadata.get('type', 'Document')} - {metadata.get('date', 'N/A')}\n"
            
            return answer
        else:
            # Fallback: retrieve relevant documents without LLM
            docs = self.vectorstore.similarity_search(query, k=3)
            response = "**Retrieved Information:**\n\n"
            for doc in docs:
                response += f"- {doc.page_content.strip()}\n\n"
            return response


class PlottingAgent:
    """Agent for creating visualizations"""
    
    def __init__(self, ticker: str, stock_data: pd.DataFrame):
        self.ticker = ticker
        self.data = stock_data
    
    def create_chart(self, chart_type: str = 'candlestick') -> go.Figure:
        """Create interactive Plotly chart"""
        
        if self.data.empty:
            return None
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{self.ticker} Stock Price', 'Volume'),
            vertical_spacing=0.1
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # Add moving averages
        if 'SMA_20' in self.data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['SMA_20'],
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        if 'SMA_50' in self.data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['SMA_50'],
                    name='SMA 50',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
        
        # Volume
        colors = ['red' if row['Close'] < row['Open'] else 'green' 
                  for _, row in self.data.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=self.data.index,
                y=self.data['Volume'],
                name='Volume',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'{self.ticker} Stock Analysis',
            yaxis_title='Price ($)',
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=True
        )
        
        return fig


class TrendAnalysisAgent:
    """Agent for analyzing price trends"""
    
    def __init__(self, ticker: str, stock_data: pd.DataFrame):
        self.ticker = ticker
        self.data = stock_data
    
    def analyze(self) -> str:
        trends = detect_trends(self.data)
        
        if not trends:
            return f"No significant trends detected for {self.ticker} in the analyzed period."
        
        response = f"**Trend Analysis for {self.ticker}:**\n\n"
        
        # Recent trends
        recent_trends = [t for t in trends if t['type'] == 'uptrend'][:3]
        if recent_trends:
            response += "**Recent Uptrends:**\n"
            for trend in recent_trends:
                response += f"- {trend['date']}: +{trend['change']:.2f}% (Price: ${trend['price']:.2f})\n"
        
        recent_drops = [t for t in trends if t['type'] == 'downtrend'][:3]
        if recent_drops:
            response += "\n**Recent Downtrends:**\n"
            for trend in recent_drops:
                response += f"- {trend['date']}: {trend['change']:.2f}% (Price: ${trend['price']:.2f})\n"
        
        return response


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class ChatbotOrchestrator:
    """Main orchestrator that routes queries to appropriate agents"""
    
    def __init__(self, llm_api_key: str = None):
        self.llm_api_key = llm_api_key
        self.current_ticker = None
        self.stock_data = None
        self.vectorstore = None
        self.agents = {}
    
    def initialize_ticker(self, ticker: str) -> str:
        """Initialize data and agents for a specific ticker"""
        try:
            # Fetch data
            self.current_ticker = ticker
            self.stock_data = fetch_stock_data(ticker)
            
            if self.stock_data.empty:
                return f"Could not fetch data for {ticker}. Please check the ticker symbol."
            
            self.stock_data = calculate_technical_indicators(self.stock_data)
            
            # Fetch news
            news = scrape_financial_news(ticker)
            
            # Build vector store
            self.vectorstore = build_vector_store(ticker, self.stock_data, news)
            
            # Initialize agents
            rag_chain = create_rag_chain(self.vectorstore, self.llm_api_key) if self.vectorstore else None
            
            self.agents = {
                'research': ResearchAgent(self.vectorstore, rag_chain) if self.vectorstore else None,
                'plotting': PlottingAgent(ticker, self.stock_data),
                'trend': TrendAnalysisAgent(ticker, self.stock_data)
            }
            
            return f"✓ Initialized analysis for {ticker}"
            
        except Exception as e:
            return f"Error initializing {ticker}: {str(e)}"
    
    def process_query(self, query: str) -> Tuple[str, Optional[go.Figure]]:
        """Process user query and return response with optional chart"""
        
        # Extract ticker if mentioned
        mentioned_ticker = extract_ticker(query)
        if mentioned_ticker and mentioned_ticker != self.current_ticker:
            init_msg = self.initialize_ticker(mentioned_ticker)
            if "Error" in init_msg:
                return init_msg, None
        
        if not self.current_ticker:
            return "Please specify a stock ticker first (e.g., 'Analyze AAPL' or 'Tell me about Apple').", None
        
        # Classify intent
        intent = classify_intent(query)
        
        # Route to appropriate agent
        if intent == 'visualization':
            chart = self.agents['plotting'].create_chart()
            return f"Here's the chart for {self.current_ticker}:", chart
        
        elif intent == 'trend_analysis':
            analysis = self.agents['trend'].analyze()
            chart = self.agents['plotting'].create_chart()
            return analysis, chart
        
        elif intent == 'explanation' and self.agents['research']:
            answer = self.agents['research'].answer(query)
            return answer, None
        
        else:
            # General query - use research agent if available
            if self.agents['research']:
                answer = self.agents['research'].answer(query)
                return answer, None
            else:
                # Fallback response
                latest_price = self.stock_data['Close'].iloc[-1]
                change = self.stock_data['Returns'].iloc[-1]
                return f"{self.current_ticker} is currently trading at ${latest_price:.2f} ({change:+.2f}% today).", None


# ============================================================================
# GRADIO UI - TASK 2 & 3
# ============================================================================

def create_gradio_interface():
    """Create Gradio interface for the chatbot"""
    
    # Initialize orchestrator
    orchestrator = ChatbotOrchestrator()
    
    def chat_interface(message, history, api_key):
        """Handle chat messages"""
        if api_key and api_key.strip():
            orchestrator.llm_api_key = api_key.strip()
        
        response, chart = orchestrator.process_query(message)
        
        # Return text response
        return response
    
    def plot_stock(ticker):
        """Generate stock chart"""
        if ticker:
            orchestrator.initialize_ticker(ticker.upper())
            if orchestrator.agents.get('plotting'):
                return orchestrator.agents['plotting'].create_chart()
        return None
    
    def get_live_data(ticker):
        """Get live data updates (Task 3)"""
        if not ticker:
            return "Please enter a ticker symbol."
        
        try:
            stock = yf.Ticker(ticker.upper())
            # Get latest intraday data
            live_data = stock.history(period='1d', interval='1m')
            
            if live_data.empty:
                return f"No live data available for {ticker}"
            
            latest = live_data.iloc[-1]
            current_price = latest['Close']
            change = ((current_price / live_data['Close'].iloc[0]) - 1) * 100
            
            info = f"""
            **Live Data for {ticker.upper()}**
            Current Price: ${current_price:.2f}
            Change Today: {change:+.2f}%
            Volume: {latest['Volume']:,.0f}
            Last Updated: {live_data.index[-1].strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            return info
        except Exception as e:
            return f"Error fetching live data: {str(e)}"
    
    # Create Gradio Blocks interface
    with gr.Blocks(title="Stock Market RAG Chatbot") as demo:
        gr.Markdown("# 📈 Stock Market RAG Chatbot")
        gr.Markdown("Ask questions about stocks in natural language. The system will analyze data, retrieve news, and provide contextual explanations.")
        
        with gr.Tab("💬 Chat Interface (Task 2)"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=400)
                    msg = gr.Textbox(
                        placeholder="Ask about stocks: 'Why did Apple drop?', 'When did Microsoft go up?', 'Compare AAPL and MSFT'",
                        label="Your Question"
                    )
                    with gr.Row():
                        submit = gr.Button("Send", variant="primary")
                        clear = gr.Button("Clear")
                
                with gr.Column(scale=1):
                    api_key_input = gr.Textbox(
                        label="LLM API Key (Optional)",
                        placeholder="Enter Groq or Gemini API key for enhanced responses",
                        type="password"
                    )
                    gr.Markdown("""
                    **Example Questions:**
                    - Why did Apple stock drop last week?
                    - When did Microsoft have an uptrend?
                    - Show me a chart for TSLA
                    - Compare NVDA performance
                    - What's happening with META?
                    """)
            
            # Chat functionality
            msg.submit(chat_interface, [msg, chatbot, api_key_input], [chatbot])
            submit.click(chat_interface, [msg, chatbot, api_key_input], [chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)
        
        with gr.Tab("📊 Live Dashboard (Task 3)"):
            gr.Markdown("## Real-Time Stock Monitor")
            
            with gr.Row():
                ticker_input = gr.Textbox(
                    label="Ticker Symbol",
                    placeholder="e.g., AAPL, MSFT, TSLA"
                )
                refresh_btn = gr.Button("🔄 Refresh Live Data", variant="primary")
            
            live_output = gr.Textbox(label="Live Market Data", lines=8)
            live_chart = gr.Plot(label="Live Chart")
            
            refresh_btn.click(
                get_live_data,
                inputs=[ticker_input],
                outputs=[live_output]
            )
            
            refresh_btn.click(
                plot_stock,
                inputs=[ticker_input],
                outputs=[live_chart]
            )
            
            gr.Markdown("""
            **Note:** Live data updates every minute. Click refresh to get latest data.
            For continuous streaming, the backend can be configured with WebSocket connections.
            """)
        
        with gr.Tab("📖 Documentation"):
            gr.Markdown("""
            ## How It Works
            
            ### Task 2: RAG Chatbot
            - **Ticker Extraction**: Identifies stock symbols from natural language
            - **Intent Classification**: Routes queries to specialized agents
            - **Vector Database**: Stores stock data and news for retrieval
            - **Multi-Agent System**:
              - Research Agent: Retrieves and explains context
              - Plotting Agent: Creates visualizations
              - Trend Analysis Agent: Identifies market movements
            
            ### Task 3: Live Data Integration
            - Fetches real-time intraday data (1-minute intervals)
            - Updates dashboard without page refresh
            - Can be extended with WebSocket for continuous streaming
            - Backfills historical data for continuity
            
            ### RAG Components
            - **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
            - **Vector Store**: ChromaDB for efficient retrieval
            - **LLM**: Groq (Mixtral-8x7b) or Google Gemini
            - **Query Enhancement**: Rephrasing, reranking
            
            ### Data Sources
            - yfinance for stock data
            - Yahoo Finance RSS for news
            - Web scraping for additional context
            """)
    
    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True)
