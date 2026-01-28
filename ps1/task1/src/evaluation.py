import numpy as np
from prophet import Prophet
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def rolling_validation(df, train_window=180, horizon=7):
    """
    Rolling-origin evaluation using expanding windows.
    Evaluates point forecast accuracy.
    """
    preds, actuals = [], []

    for start in range(0, len(df) - train_window - horizon, horizon):
        train = df.iloc[start : start + train_window]
        test = df.iloc[start + train_window : start + train_window + horizon]

        model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        model.fit(train)

        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)

        y_pred = np.exp(forecast["yhat"].iloc[-horizon:])
        y_true = np.exp(test["y"])

        preds.extend(y_pred)
        actuals.extend(y_true)

    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "Evaluations": len(actuals)
    }

import pandas as pd

def interval_coverage(df, model):
    """
    Empirical coverage of Prophet prediction intervals
    using in-sample predictions.
    """

    # Try to infer frequency, fall back safely
    freq = pd.infer_freq(df["ds"])
    if freq is None:
        freq = "D"   # safe default

    # In-sample forecast only
    future = model.make_future_dataframe(
        periods=0,
        freq=freq
    )

    forecast = model.predict(future)

    # Align forecast with actuals
    merged = pd.merge(
        df[["ds", "y"]],
        forecast[["ds", "yhat_lower", "yhat_upper"]],
        on="ds",
        how="inner"
    )

    if merged.empty:
        return float("nan")

    coverage = (
        (merged["y"] >= merged["yhat_lower"]) &
        (merged["y"] <= merged["yhat_upper"])
    ).mean()

    return float(coverage)

