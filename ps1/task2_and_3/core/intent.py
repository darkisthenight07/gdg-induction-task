import re
from core.config import SUPPORTED_TICKERS

def extract_ticker(query: str):
    match = re.search(r"\b[A-Z]{1,5}\b", query)
    if match:
        return match.group()

    q = query.lower()
    for name, ticker in SUPPORTED_TICKERS.items():
        if name in q:
            return ticker
    return None

def classify_intent(query: str):
    q = query.lower()
    if any(w in q for w in ["why", "reason", "explain", "cause"]):
        return "explanation"
    if any(w in q for w in ["when", "trend", "up", "down"]):
        return "trend"
    return "general"
