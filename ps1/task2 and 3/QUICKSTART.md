# 🚀 Quick Start Guide

## Option 1: Demo Version (No API Key Needed) - Fastest

```bash
# Install dependencies
pip install gradio yfinance pandas numpy plotly

# Run demo
python demo_app.py
```

Opens at `http://localhost:7860`

**Features in Demo:**
- ✅ Stock data analysis
- ✅ Trend detection with dates
- ✅ Interactive charts
- ✅ Live data integration
- ✅ Intent-based routing
- ❌ No LLM (rule-based responses)

## Option 2: Full Version (With LLM) - Best Experience

### Step 1: Get API Key (Choose One)

**Groq (Recommended - Free & Fast)**
1. Go to https://console.groq.com
2. Sign up / Log in
3. Go to "API Keys"
4. Create new key
5. Copy the key

**Google Gemini (Free)**
1. Go to https://makersuite.google.com/app/apikey
2. Create API key
3. Copy the key

### Step 2: Setup

```bash
# Run setup script
bash setup.sh

# Or manual setup:
pip install -r requirements.txt
cp .env.example .env
nano .env  # Add your API key
```

### Step 3: Add API Key

Edit `.env` file:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxx
```

### Step 4: Run

```bash
python app.py
```

## 📝 Usage Examples

### Basic Queries
```
"Tell me about Apple"
"What's MSFT doing?"
"Show me Tesla chart"
```

### Task 2: Contextual Analysis
```
"Why did Apple stock drop?"
→ System identifies AAPL, retrieves news, explains cause with dates

"When did Microsoft go up?"
→ Analyzes MSFT data, shows uptrend dates with percentages

"Compare NVDA and AMD"
→ Fetches both, analyzes trends, shows comparative charts
```

### Task 3: Live Data
```
Go to "Live Dashboard" tab
Enter: AAPL
Click: Refresh Live Data
→ Shows real-time price, updates every minute
```

## 🔧 Troubleshooting

### "Module not found"
```bash
pip install -r requirements.txt
```

### "API key invalid"
- Check .env file has correct key
- Make sure no spaces around `=`
- Try regenerating key

### "No data for ticker"
- Check ticker symbol (AAPL not Apple)
- Check market hours
- Try different ticker

### "Gradio won't start"
```bash
pip install --upgrade gradio
python demo_app.py  # Try demo first
```

## 📊 What Each File Does

```
app.py              # Full version with LLM
demo_app.py         # Demo version (no API key)
requirements.txt    # Dependencies
.env                # Your API keys (YOU create this)
.env.example        # Template for .env
README.md           # Full documentation
TECHNICAL_GUIDE.md  # Explains how it works
setup.sh            # Automated setup script
```

## 🎯 For Evaluation

**What You Need to Explain:**

1. **RAG Pipeline:**
   - How embeddings work
   - Why vector database
   - Retrieval → Augment → Generate flow

2. **Multi-Agent System:**
   - Why separate agents
   - How orchestrator routes queries
   - Each agent's responsibility

3. **Intent Classification:**
   - Not just keywords
   - Semantic understanding
   - Pattern matching logic

4. **Live Data (Task 3):**
   - Background threading
   - Auto-refresh mechanism
   - Data continuity strategy

5. **Technical Indicators:**
   - What SMA, RSI mean
   - How to detect trends
   - Why 5% threshold

**What You Built:**

✅ Task 2 Complete:
- Natural language interface ✓
- Ticker extraction ✓
- Intent classification ✓
- RAG with vector DB ✓
- Multi-agent architecture ✓
- News integration ✓
- Contextual explanations ✓

✅ Task 3 Complete:
- Live data ingestion ✓
- Real-time updates ✓
- Auto-refresh ✓
- Intraday data (1-min) ✓
- Historical backfilling ✓

## 💡 Tips for Demo

1. **Start with Demo:**
   - Show it works without API key
   - Explain the logic clearly
   - Then show full version

2. **Explain Trade-offs:**
   - Demo: Fast, simple, rule-based
   - Full: Slower, smarter, LLM-based

3. **Show Understanding:**
   - Don't just run code
   - Explain WHY each part exists
   - Discuss alternatives

4. **Be Honest:**
   - "This could be improved by..."
   - "I chose X over Y because..."
   - "Next step would be..."

## 🎓 Key Concepts to Master

Before demo, make sure you can explain:
- ✅ What is RAG and why use it
- ✅ How embeddings enable semantic search
- ✅ Why multi-agent vs monolithic
- ✅ How intent classification works
- ✅ Live data streaming approach
- ✅ Vector database purpose
- ✅ LangChain orchestration

Read `TECHNICAL_GUIDE.md` for detailed explanations!

## 📚 Quick Commands Reference

```bash
# Demo (no API key)
python demo_app.py

# Full version
python app.py

# Install specific package
pip install package_name

# Check installation
python -c "import gradio; print('OK')"

# Activate venv (if using)
source venv/bin/activate
```

---

**Need Help?** Check TECHNICAL_GUIDE.md for concept explanations!
