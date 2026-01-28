import pandas as pd
import numpy as np

def load_stock_csv(file):
    df = pd.read_csv(file)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Try to detect date column
    date_candidates = ["Date", "date", "Datetime", "datetime"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col is None:
        raise ValueError("No Date column found in CSV")

    # Try to detect close/price column
    price_candidates = ["Close", "close", "Adj Close", "Adj_Close", "price", "Price"]
    price_col = next((c for c in price_candidates if c in df.columns), None)
    if price_col is None:
        raise ValueError("No price/Close column found in CSV")

    df = df[[date_col, price_col]].copy()
    df.columns = ["ds", "y"]

    # Parse date
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

    # Convert price safely
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # Drop bad rows
    df = df.dropna()

    # Safety check
    if len(df) < 30:
        raise ValueError("Not enough valid rows after cleaning")

    # Log transform if strictly positive
    log_transform = False
    if (df["y"] > 0).all():
        df["y"] = np.log(df["y"])
        log_transform = True

    return df, log_transform
