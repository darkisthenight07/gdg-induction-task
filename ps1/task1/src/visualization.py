import plotly.graph_objects as go
import numpy as np

def plot_forecast(df, forecast):
    """
    Plots historical prices, forecast, confidence interval,
    and annotates recent volatility.
    """
    fig = go.Figure()

    # Historical prices (back to price space)
    fig.add_trace(go.Scatter(
        x=df["ds"],
        y=np.exp(df["y"]),
        name="Historical",
        line=dict(color="blue")
    ))

    # Forecast mean
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        name="Forecast",
        line=dict(color="orange")
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_lower"],
        fill="tonexty",
        mode="lines",
        line=dict(width=0),
        name="Confidence Interval",
        opacity=0.3
    ))

    # Volatility annotation
    returns = np.diff(df["y"].values)
    recent_vol = np.std(returns[-20:])

    fig.add_annotation(
        text=f"Recent 20-day volatility (log-returns): {recent_vol:.4f}",
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.95,
        showarrow=False
    )

    fig.update_layout(
        title="Stock Price Forecast with Uncertainty",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    return fig
