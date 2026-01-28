import yfinance as yf

df = yf.download("AAPL", start="2018-01-01", end="2024-01-01")
df.reset_index().to_csv("AAPL.csv", index=False)
