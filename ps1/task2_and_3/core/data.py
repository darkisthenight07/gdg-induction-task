import yfinance as yf
import pandas as pd

def fetch_historical(ticker, period="6mo"):
    df = yf.Ticker(ticker).history(period=period)
    if df.empty:
        raise ValueError("No historical data")
    return df

def fetch_live_price(ticker):
    df = yf.Ticker(ticker).history(period="1d", interval="1m")
    if df.empty:
        raise ValueError("No live data")
    return float(df["Close"].iloc[-1])

