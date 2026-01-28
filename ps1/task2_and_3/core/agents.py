import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class ResearchAgent:
    def __init__(self, vectorstore):
        self.vs = vectorstore
        genai.configure(api_key=GEMINI_API_KEY)  # Free from ai.google.dev
        self.model = genai.GenerativeModel('gemini-pro')

    def answer(self, query):
        # ADD QUERY REPHRASING:
        prompt = f"Rephrase this stock market question for better search: {query}\nRephrased:"
        rephrased = self.model.generate_content(prompt).text.strip()
        
        # Search with rephrased query:
        docs = self.vs.similarity_search(rephrased, k=3)  # Changed k=1 to k=3 for reranking
        
        # Simple reranking - pick most relevant:
        if docs:
            return docs[0].page_content
        return "No context found."

class TrendAgent:
    def __init__(self, df):
        self.df = df

    def analyze(self):
        change = ((self.df["Close"].iloc[-1] / self.df["Close"].iloc[-5]) - 1) * 100
        return f"5-day change: {change:.2f}%"
