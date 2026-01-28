import pandas as pd

def add_indicators(df: pd.DataFrame):
    df = df.copy()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df

def summarize_trend(df):
    if df["SMA_20"].iloc[-1] > df["SMA_50"].iloc[-1]:
        return "Uptrend 📈"
    return "Downtrend 📉"

def percent_change(df, days=5):
    return round(((df["Close"].iloc[-1] / df["Close"].iloc[-days]) - 1) * 100, 2)
