from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def build_vectorstore(ticker, df):
    docs = []

    latest = df.iloc[-1]
    summary = f"""
    Stock: {ticker}
    Price: {latest['Close']:.2f}
    SMA20: {latest['SMA_20']:.2f}
    SMA50: {latest['SMA_50']:.2f}
    RSI: {latest['RSI']:.2f}
    """

    docs.append(Document(page_content=summary, metadata={"type": "summary"}))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma.from_documents(docs, embeddings)
