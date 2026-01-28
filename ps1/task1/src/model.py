import numpy as np
from prophet import Prophet

"""
Prophet is used as a probabilistic time-series model that decomposes
price into trend + seasonal components. Unlike point predictors,
Prophet outputs a posterior predictive distribution, allowing us
to explicitly model uncertainty via confidence intervals.
"""

def train_prophet(df, interval_width=0.8):
    """
    Trains a Prophet model with weekly and yearly seasonality.
    """
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=interval_width
    )
    model.fit(df)
    return model

def make_forecast(model, periods=7, log_transform=True):
    """
    Generates future forecast and converts back to price space if needed.
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    if log_transform:
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            forecast[col] = np.exp(forecast[col])

    return forecast
