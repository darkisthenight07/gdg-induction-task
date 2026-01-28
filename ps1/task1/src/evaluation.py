import numpy as np
from prophet import Prophet
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

def interval_coverage(df, model, horizon=7):
    """
    Measures how often true prices fall within prediction intervals.
    """
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)

    y_true = np.exp(df["y"].iloc[-horizon:])
    lower = forecast["yhat_lower"].iloc[-horizon:]
    upper = forecast["yhat_upper"].iloc[-horizon:]

    coverage = ((y_true >= lower) & (y_true <= upper)).mean()
    return round(float(coverage), 3)
