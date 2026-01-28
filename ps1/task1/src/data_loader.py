import pandas as pd
import numpy as np

def load_stock_csv(file, log_transform=True):
    """
    Loads stock CSV and prepares data for Prophet.
    Optionally applies log-transform to stabilize variance.
    """
    df = pd.read_csv(file)

    required_cols = {"Date", "Close"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV must contain Date and Close columns")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    y = df["Close"].astype(float)
    if log_transform:
        y = np.log(y)

    prophet_df = pd.DataFrame({
        "ds": df["Date"],
        "y": y
    }).dropna()

    if len(prophet_df) < 100:
        raise ValueError("Not enough data (minimum 100 rows required)")

    return prophet_df, log_transform
