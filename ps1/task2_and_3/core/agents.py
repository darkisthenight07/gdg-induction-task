class ResearchAgent:
    def __init__(self, vectorstore):
        self.vs = vectorstore

    def answer(self, query):
        docs = self.vs.similarity_search(query, k=1)
        return docs[0].page_content if docs else "No context found."

class TrendAgent:
    def __init__(self, df):
        self.df = df

    def analyze(self):
        change = ((self.df["Close"].iloc[-1] / self.df["Close"].iloc[-5]) - 1) * 100
        return f"5-day change: {change:.2f}%"
