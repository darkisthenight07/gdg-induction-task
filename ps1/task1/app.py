import gradio as gr

from src.data_loader import load_stock_csv
from src.model import train_prophet, make_forecast
from src.visualization import plot_forecast
from src.evaluation import rolling_validation, interval_coverage
import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def run_pipeline(file, interval_width, show_components):
    try:
        df, log_transform = load_stock_csv(file)
    except Exception as e:
        return f"CSV error: {str(e)}", None, None

    df, log_transform = load_stock_csv(file)

    model = train_prophet(df, interval_width)
    forecast = make_forecast(model, periods=7, log_transform=log_transform)

    main_fig = plot_forecast(df, forecast)

    components_fig = None
    if show_components:
        components_fig = model.plot_components(forecast)

    metrics = rolling_validation(df)
    coverage = interval_coverage(df, model)

    metrics_text = (
        "Backtest Performance (Rolling Validation)\n"
        f"MAE: {metrics['MAE']}\n"
        f"RMSE: {metrics['RMSE']}\n"
        f"Interval Coverage (last 7 days): {coverage}\n"
        f"Total Predictions Evaluated: {metrics['Evaluations']}"
    )

    return main_fig, components_fig, metrics_text

with gr.Blocks() as demo:
    gr.Markdown("## 📈 Predictive Core — Probabilistic Stock Forecasting Engine")

    gr.Markdown(
        """
        This tool forecasts future stock prices while explicitly modeling uncertainty.
        The shaded region represents the confidence interval derived from the model's
        posterior predictive distribution.
        """
    )

    file = gr.File(label="Upload Stock CSV")
    interval = gr.Slider(0.6, 0.95, value=0.8, label="Confidence Interval Width")
    show_components = gr.Checkbox(
        label="Show Model Components (Trend & Seasonality)",
        value=False
    )

    btn = gr.Button("Run Forecast")

    plot = gr.Plot(label="Forecast")
    components_plot = gr.Plot(label="Model Components")
    metrics = gr.Textbox(label="Evaluation Metrics", lines=6)

    btn.click(
        run_pipeline,
        inputs=[file, interval, show_components],
        outputs=[plot, components_plot, metrics]
    )

demo.launch()
