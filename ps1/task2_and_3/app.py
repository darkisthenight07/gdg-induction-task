import gradio as gr
from core.orchestrator import Orchestrator
from core.data import fetch_live_price

orch = Orchestrator()

def chat(query):
    return orch.handle(query)

def live(ticker):
    price = fetch_live_price(ticker)
    return f"{ticker}: ${price:.2f}"


with gr.Blocks() as app:
    gr.Markdown("# 📈 Market Intelligence Chatbot")

    q = gr.Textbox(label="Ask a market question")
    out = gr.Textbox()
    q.submit(chat, q, out)

    gr.Markdown("## 🔴 Live Price")
    t = gr.Textbox("AAPL")
    l = gr.Textbox()
    gr.Button("Fetch").click(live, t, l)

app.launch()
