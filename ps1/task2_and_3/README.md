# 📈 Market Intelligence Chatbot

> **GDG Induction Task - Problem Statement 1: Tasks 2 & 3**  
> An intelligent, context-aware chatbot for real-time stock market analysis using RAG, multi-agent architecture, and live data integration.

---

## 🎯 Project Overview

This project implements a sophisticated financial analysis chatbot that combines:
- **Natural Language Understanding**: Semantic intent classification and ticker extraction
- **RAG Architecture**: Vector-based retrieval for contextual responses
- **Multi-Agent System**: Specialized agents for research and trend analysis
- **Live Data Integration**: Real-time price tracking and updates
- **Technical Analysis**: Automated indicator calculation (SMA, RSI)

### ✨ Key Features

#### Task 2: Analytical Chatbot (RAG & Explanation)
- ✅ **Contextual Query Processing**: Natural language understanding for stock queries
- ✅ **Ticker Identification**: Intelligent extraction from company names or symbols
- ✅ **Multi-Agent Architecture**: 
  - `ResearchAgent`: RAG-based contextual answers
  - `TrendAgent`: Technical analysis and trend detection
- ✅ **Intent Classification**: Semantic understanding of user queries
- ✅ **Vector Database**: ChromaDB with HuggingFace embeddings
- ✅ **Technical Indicators**: SMA (20, 50), RSI calculation

#### Task 3: Real-Time Intelligence (Live Data Integration)
- ✅ **Live Price Fetching**: Real-time market data via yfinance API
- ✅ **Interactive Dashboard**: Gradio-based UI for instant updates
- ✅ **Data Continuity**: Historical + live data integration
- ✅ **Dynamic Updates**: 1-minute interval price refresh capability

---

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────────────┐
│                      User Interface                     │
│                    (Gradio Web App)                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator                         │
│  • Query Routing    • Ticker Management                │
│  • Agent Selection  • Data Loading                     │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│    Intent    │ │  Data Layer  │ │    Agents    │
│ Classifier   │ │              │ │              │
│              │ │ • Historical │ │ • Research   │
│ • Extraction │ │ • Live Price │ │ • Trend      │
│ • Intent     │ │ • Indicators │ │              │
└──────────────┘ └──────────────┘ └──────┬───────┘
                                         │
                                         ▼
                                  ┌──────────────┐
                                  │  RAG System  │
                                  │              │
                                  │ • Vector DB  │
                                  │ • Embeddings │
                                  └──────────────┘
```

### Module Breakdown

| Module | Purpose | Key Components |
|--------|---------|----------------|
| `app.py` | Entry point & UI | Gradio interface, chat handler |
| `orchestrator.py` | Request routing | Query processing, agent coordination |
| `intent.py` | NLP processing | Ticker extraction, intent classification |
| `agents.py` | Multi-agent system | ResearchAgent, TrendAgent |
| `rag.py` | Retrieval system | Vector store, embeddings |
| `data.py` | Data fetching | Historical & live data from yfinance |
| `analysis.py` | Technical analysis | SMA, RSI, trend detection |
| `config.py` | Configuration | Supported tickers, parameters |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for data fetching)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/darkisthenight07/gdg-induction-task.git
cd gdg-induction-task/ps1/task2_and_3
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Access the interface**
   - Open your browser to the URL shown in terminal (typically `http://127.0.0.1:7860`)

---

## 💡 Usage Examples

### Query Examples

#### 1. Contextual Analysis
```
User: "Why did Apple stock drop?"
System: 
  → Identifies ticker: AAPL
  → Intent: explanation
  → Retrieves context from vector DB
  → Returns: Technical analysis with RSI, SMA data
```

#### 2. Trend Analysis
```
User: "When did Microsoft go up?"
System:
  → Identifies ticker: MSFT
  → Intent: trend
  → Analyzes SMA crossovers
  → Returns: "Uptrend 📈" or "Downtrend 📉"
```

#### 3. Price Change
```
User: "How is Tesla performing?"
System:
  → Identifies ticker: TSLA
  → Intent: general
  → Calculates 5-day change
  → Returns: "TSLA: +5.32% (5 days)"
```

#### 4. Live Price
```
User: Enters "NVDA" in live price section
System:
  → Fetches real-time price
  → Returns: "NVDA: $875.43"
```

### Supported Stocks

The system supports the following major tech stocks:

| Company | Ticker | Keyword |
|---------|--------|---------|
| Apple | AAPL | "apple" |
| Microsoft | MSFT | "microsoft" |
| Google | GOOGL | "google" |
| Amazon | AMZN | "amazon" |
| Tesla | TSLA | "tesla" |
| Meta | META | "meta" |
| NVIDIA | NVDA | "nvidia" |

---

## 🧠 Technical Implementation

### 1. Intent Classification

The system uses keyword-based semantic analysis to classify queries:

```python
Intent Types:
- explanation: "why", "reason", "explain", "cause"
- trend: "when", "trend", "up", "down"
- general: Default fallback
```

### 2. RAG Pipeline

**Vector Store Construction:**
- Documents: Stock summaries with latest indicators
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector DB: ChromaDB for fast similarity search
- Retrieval: Top-k similarity search (k=1)

**Document Format:**
```
Stock: AAPL
Price: 175.43
SMA20: 172.15
SMA50: 168.92
RSI: 58.73
```

### 3. Technical Indicators

#### Simple Moving Averages (SMA)
- **SMA-20**: 20-day average (short-term trend)
- **SMA-50**: 50-day average (long-term trend)
- **Crossover Logic**: SMA-20 > SMA-50 = Uptrend

#### Relative Strength Index (RSI)
- **Calculation**: 14-period RSI
- **Interpretation**: 
  - RSI > 70: Overbought
  - RSI < 30: Oversold
  - 30-70: Neutral zone

### 4. Data Pipeline

```python
Historical Data Flow:
yfinance API → Pandas DataFrame → Add Indicators → Vector Store

Live Data Flow:
yfinance (1m interval) → Latest close price → UI Update
```

---

## 📊 Agent Architecture

### ResearchAgent
- **Purpose**: Answer contextual questions using RAG
- **Process**:
  1. Receives user query
  2. Performs similarity search in vector store
  3. Returns most relevant context
- **Use Case**: "Why did stock X move?"

### TrendAgent
- **Purpose**: Analyze price trends and momentum
- **Process**:
  1. Accesses stock DataFrame
  2. Calculates percentage changes
  3. Identifies trend direction
- **Use Case**: "What's the trend for stock X?"

---

## 🎨 User Interface

### Chat Interface
- **Input**: Natural language query
- **Output**: Contextual response with technical data
- **Features**: 
  - Auto-submit on Enter
  - Real-time processing
  - Error handling

### Live Price Monitor
- **Input**: Stock ticker symbol
- **Output**: Current market price
- **Update**: Manual refresh via button
- **Potential**: Can be extended to WebSocket auto-refresh

---

## 🔧 Configuration

### Modifying Supported Tickers

Edit `config.py`:
```python
SUPPORTED_TICKERS = {
    "company_name": "TICKER",
    "netflix": "NFLX",  # Add new stock
}
```

### Adjusting Data Parameters

```python
HISTORICAL_PERIOD = "1y"  # 1 year of historical data
LIVE_INTERVAL = "1m"      # 1-minute live data resolution
```

### Customizing Indicators

In `analysis.py`, modify rolling windows:
```python
df["SMA_20"] = df["Close"].rolling(20).mean()  # Change to 30
df["SMA_50"] = df["Close"].rolling(50).mean()  # Change to 100
```
---