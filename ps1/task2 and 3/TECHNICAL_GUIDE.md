# Technical Documentation: Understanding the Implementation

## 🧠 Core Concepts Explained

### 1. RAG (Retrieval Augmented Generation)

**What is it?**
RAG combines retrieval-based and generative AI. Instead of relying solely on an LLM's training data, we retrieve relevant information first, then use it as context for generation.

**Our Implementation:**

```
User Query: "Why did Apple drop?"
     ↓
Step 1: RETRIEVE
   - Search vector database for relevant info about AAPL
   - Find: news articles, price drops, dates
     ↓
Step 2: AUGMENT
   - Add retrieved context to the prompt
   - Include specific dates, prices, news headlines
     ↓
Step 3: GENERATE
   - LLM synthesizes explanation using context
   - Cites sources from retrieved documents
```

**Why RAG?**
- ✅ Reduces hallucination (LLM has real data)
- ✅ Provides citations
- ✅ Works with up-to-date information
- ✅ More accurate than pure LLM

### 2. Vector Database & Embeddings

**The Problem:**
How do we find relevant information from thousands of documents?

**The Solution: Semantic Search**

```python
# Traditional keyword search
"Apple stock drop" → finds documents with exact words

# Semantic search (embeddings)
"Why did Apple decline?" → finds documents about:
  - Apple price decreases
  - AAPL downtrends
  - Negative Apple news
  # (same meaning, different words!)
```

**How it Works:**

1. **Text → Numbers (Embeddings)**
   ```python
   "Apple stock dropped 5%" → [0.23, -0.45, 0.67, ..., 0.12]
   # 384 numbers representing the meaning
   ```

2. **Store in Vector Database**
   ```
   ChromaDB stores:
   - Document text
   - Embedding vector
   - Metadata (date, source, type)
   ```

3. **Similarity Search**
   ```python
   query = "Why did Apple fall?"
   query_embedding = embed(query)
   
   # Find documents with similar embeddings
   similar_docs = vectordb.similarity_search(query_embedding, k=5)
   # Returns top 5 most relevant documents
   ```

**Our Embedding Model:**
- `sentence-transformers/all-MiniLM-L6-v2`
- Converts text → 384-dimensional vectors
- Fast, efficient, good for financial text

### 3. Multi-Agent Architecture

**Why Agents?**
Different tasks need different skills. Instead of one monolithic system, we have specialized agents.

**Our Agents:**

```
                    Orchestrator
                         |
        ┌────────────────┼────────────────┐
        ↓                ↓                 ↓
  Research Agent    Plotting Agent    Trend Agent
  
  Handles:          Handles:          Handles:
  - "Why?"          - "Show chart"    - "When?"
  - "Explain"       - "Plot"          - "Trend analysis"
  - Uses RAG        - Creates viz     - Finds patterns
```

**Agent Implementation:**

```python
class ResearchAgent:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
    
    def answer(self, query):
        # 1. Retrieve relevant docs
        docs = self.vectorstore.similarity_search(query, k=5)
        
        # 2. Create context from docs
        context = "\n".join([doc.page_content for doc in docs])
        
        # 3. Generate answer with LLM
        prompt = f"Context: {context}\n\nQuestion: {query}"
        answer = self.llm(prompt)
        
        return answer

class PlottingAgent:
    def create_chart(self, ticker, data):
        # Creates Plotly visualization
        return plotly_chart

class TrendAnalysisAgent:
    def analyze(self, data):
        # Detects significant price movements
        # Returns dates and percentages
        return trends
```

**Orchestrator Routes Queries:**

```python
def process_query(query):
    intent = classify_intent(query)
    
    if intent == 'explanation':
        return research_agent.answer(query)
    elif intent == 'visualization':
        return plotting_agent.create_chart()
    elif intent == 'trend_analysis':
        return trend_agent.analyze()
```

### 4. Intent Classification

**Not Just Keywords!**

❌ **Bad (Keyword Matching):**
```python
if "why" in query:
    return explanation()
```

✅ **Good (Semantic Understanding):**
```python
def classify_intent(query):
    query_lower = query.lower()
    
    # Explanation intent
    explanation_patterns = ['why', 'reason', 'cause', 'explain', 'happened']
    if any(word in query_lower for word in explanation_patterns):
        return 'explanation'
    
    # Trend analysis intent
    trend_patterns = ['when', 'date', 'time', 'trend']
    if any(word in query_lower for word in trend_patterns):
        return 'trend_analysis'
    
    # Visualization intent
    viz_patterns = ['plot', 'chart', 'graph', 'show', 'visualize']
    if any(word in query_lower for word in viz_patterns):
        return 'visualization'
    
    return 'general'
```

**Even Better: Embedding-Based Classification**
```python
# Classify by semantic similarity to intent examples
intents = {
    'explanation': ["Why did this happen?", "What caused the drop?"],
    'trend': ["When did it rise?", "Show me the trend"],
    'viz': ["Plot the chart", "Show me a graph"]
}

# Find closest intent by embedding similarity
query_emb = embed(query)
best_intent = max(intents, key=lambda i: 
    similarity(query_emb, embed(intents[i])))
```

### 5. Ticker Extraction

**Challenge:** Extract stock symbols from natural language

```python
def extract_ticker(query):
    query_lower = query.lower()
    
    # Method 1: Direct ticker mention
    ticker_pattern = r'\b[A-Z]{1,5}\b'
    matches = re.findall(ticker_pattern, query)
    if matches:
        return matches[0]  # "Check AAPL" → "AAPL"
    
    # Method 2: Company name mapping
    COMPANIES = {
        'apple': 'AAPL',
        'microsoft': 'MSFT',
        'google': 'GOOGL'
    }
    
    for company, ticker in COMPANIES.items():
        if company in query_lower:
            return ticker  # "apple stock" → "AAPL"
    
    return None
```

**Advanced: NER (Named Entity Recognition)**
```python
# Use a trained model to identify companies
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER")
entities = ner("Apple released new iPhone")
# Finds "Apple" as ORGANIZATION
```

### 6. Real-Time Data Integration (Task 3)

**Challenge:** Keep data fresh without manual refresh

**Our Approach:**

```python
class LiveDataManager:
    def __init__(self):
        self.active_tickers = []
        self.live_data = {}
        self.is_streaming = False
    
    def start_streaming(self, tickers):
        self.active_tickers = tickers
        self.is_streaming = True
        
        # Run in background thread
        threading.Thread(target=self._stream_loop, daemon=True).start()
    
    def _stream_loop(self):
        while self.is_streaming:
            for ticker in self.active_tickers:
                # Fetch latest 1-minute data
                data = yfinance.Ticker(ticker).history(
                    period='1d',
                    interval='1m'
                )
                self.live_data[ticker] = data
            
            # Update every 60 seconds
            time.sleep(60)
```

**Gradio Auto-Refresh:**
```python
with gr.Blocks() as demo:
    live_output = gr.Textbox()
    
    # Automatically refresh every minute
    demo.load(
        get_live_data,
        inputs=[ticker],
        outputs=[live_output],
        every=60  # seconds
    )
```

**Production WebSocket Approach:**
```python
import websockets
import asyncio

async def stream_prices(ticker):
    uri = f"wss://stream.example.com/{ticker}"
    async with websockets.connect(uri) as ws:
        while True:
            price = await ws.recv()
            yield json.loads(price)
```

### 7. Technical Indicators

**Why?** Identify trends programmatically

**Simple Moving Average (SMA):**
```python
# 20-day average
df['SMA_20'] = df['Close'].rolling(window=20).mean()

# When price crosses SMA = signal
if price > SMA_20:
    trend = "uptrend"
```

**Relative Strength Index (RSI):**
```python
# Measures momentum (0-100)
# > 70 = overbought (might drop)
# < 30 = oversold (might rise)

delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
```

**Detecting Significant Moves:**
```python
# 5-day percentage change
df['Change_5d'] = df['Close'].pct_change(periods=5) * 100

# Find big moves
uptrends = df[df['Change_5d'] > 5]  # >5% up
downtrends = df[df['Change_5d'] < -5]  # >5% down
```

### 8. Query Rephrasing & Reranking

**Query Rephrasing:**
```python
def rephrase_query(query, llm):
    prompt = f"""
    Rephrase this query to be more specific for searching stock data:
    
    Original: {query}
    
    Rephrased:
    """
    
    return llm(prompt)

# "Why did it drop?" → 
# "What caused the stock price to decrease?"
```

**Reranking:**
```python
def rerank_documents(query, documents, reranker):
    # Initial retrieval: 20 docs
    initial_docs = vectorstore.similarity_search(query, k=20)
    
    # Rerank by relevance
    scores = []
    for doc in initial_docs:
        score = reranker.score(query, doc.page_content)
        scores.append((score, doc))
    
    # Return top 5
    scores.sort(reverse=True)
    return [doc for score, doc in scores[:5]]
```

### 9. News Integration & Sentiment

**Multi-Source Scraping:**

```python
def scrape_news(ticker):
    news = []
    
    # Yahoo Finance RSS
    rss = feedparser.parse(f"https://finance.yahoo.com/rss/headline?s={ticker}")
    for entry in rss.entries:
        news.append({
            'title': entry.title,
            'content': entry.summary,
            'date': entry.published
        })
    
    # Google Finance (web scraping)
    url = f"https://www.google.com/finance/quote/{ticker}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract news divs...
    
    return news
```

**Sentiment Analysis:**
```python
from transformers import pipeline

sentiment = pipeline("sentiment-analysis", 
                     model="ProsusAI/finbert")

for article in news:
    result = sentiment(article['content'])
    article['sentiment'] = result[0]['label']  # POSITIVE/NEGATIVE
    article['confidence'] = result[0]['score']
```

### 10. Error Handling & Fallbacks

**Graceful Degradation:**

```python
def process_query(query):
    try:
        # Try with LLM
        if llm_available:
            return rag_chain(query)
        else:
            # Fallback: rule-based
            return simple_analysis(query)
    except Exception as e:
        # Error fallback
        return f"Error: {e}. Here's basic info: {get_stock_data(ticker)}"
```

## 🎓 Key Takeaways

1. **RAG = Retrieval + LLM**: Always ground responses in real data
2. **Embeddings**: Convert text to vectors for semantic search
3. **Agents**: Specialized modules for different tasks
4. **Intent**: Understand what user wants, not just keywords
5. **Real-time**: Background threads + auto-refresh
6. **Fallbacks**: Always have a Plan B

## 📚 Further Reading

- LangChain Documentation: https://python.langchain.com/docs/
- Vector Databases: https://www.pinecone.io/learn/vector-database/
- Sentence Transformers: https://www.sbert.net/
- Technical Analysis: https://www.investopedia.com/technical-analysis-4689657

---

**Remember:** Understanding > Implementation
You must be able to explain WHY each component exists and HOW it works!
