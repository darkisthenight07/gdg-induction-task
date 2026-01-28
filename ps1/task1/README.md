# The Predictive Core — Static Time Series Forecasting

## Overview
This project implements a probabilistic stock price forecasting engine using Facebook Prophet.
The goal is not only to predict future prices but to explicitly model uncertainty and volatility.

## Input
CSV file with daily OHLCV data.
Required columns:
- Date
- Close

## Model
We use Prophet, an additive time series model that decomposes price into:
- Trend
- Weekly seasonality
- Yearly seasonality
- Noise

Uncertainty is modeled using prediction intervals derived from the posterior distribution.

## Why Prophet?
Financial time series are noisy, non-stationary, and exhibit regime shifts.
Rather than assuming fixed statistical properties, Prophet models price as
a sum of trend, seasonality, and noise, and estimates uncertainty using
Bayesian posterior inference.

This allows us to:
- Avoid overfitting
- Quantify forecast confidence
- Perform rolling-origin backtesting

### Log-Price Modeling
Stock prices exhibit multiplicative growth and time-varying volatility.
To stabilize variance and improve trend estimation, the model is trained
on log-transformed prices. Forecasts are exponentiated back to price space
for interpretability.

## Output
- Historical price plot
- 7-day future forecast
- Confidence interval shading
- Accuracy metrics using rolling forecast validation

## Evaluation
Rolling validation is performed using:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

## How to Run
```bash
pip install -r requirements.txt
python app.py
```

## Failure Modes
- Sudden black-swan events are not predictable from historical prices
- Model assumes continuity of market regimes
- Confidence intervals widen during volatile periods