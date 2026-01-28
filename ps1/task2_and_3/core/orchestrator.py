from core.intent import extract_ticker, classify_intent
from core.data import fetch_historical
from core.analysis import add_indicators, summarize_trend, percent_change
from core.rag import build_vectorstore
from core.agents import ResearchAgent, TrendAgent
from core.config import HISTORICAL_PERIOD

class Orchestrator:
    def __init__(self):
        self.ticker = None
        self.df = None
        self.agents = {}

    def load_ticker(self, ticker):
        self.ticker = ticker
        self.df = add_indicators(fetch_historical(ticker, HISTORICAL_PERIOD))
        self.agents["research"] = ResearchAgent(build_vectorstore(ticker, self.df))
        self.agents["trend"] = TrendAgent(self.df)

    def handle(self, query):
        ticker = extract_ticker(query)
        if ticker and ticker != self.ticker:
            self.load_ticker(ticker)

        if not self.ticker:
            return "Please mention a stock."

        intent = classify_intent(query)

        if intent == "trend":
            return summarize_trend(self.df)
        if intent == "explanation":
            return self.agents["research"].answer(query)

        return f"{self.ticker}: {percent_change(self.df)}% (5 days)"
