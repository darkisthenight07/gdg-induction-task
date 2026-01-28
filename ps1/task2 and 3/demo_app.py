"""
Simplified Demo Version - Works without API keys
Quick test version of the RAG chatbot
"""

import gradio as gr
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Simple ticker mapping
TICKERS = {
    'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL',
    'amazon': 'AMZN', 'tesla': 'TSLA', 'meta': 'META',
    'nvidia': 'NVDA', 'netflix': 'NFLX'
}


def extract_ticker(query: str) -> Optional[str]:
    """Extract ticker from query"""
    query_lower = query.lower()
    
    # Direct ticker
    matches = re.findall(r'\b[A-Z]{1,5}\b', query)
    if matches:
        return matches[0]
    
    # Company name
    for name, ticker in TICKERS.items():
        if name in query_lower:
            return ticker
    
    return None


def fetch_stock_data(ticker: str) -> pd.DataFrame:
    """Fetch stock data"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1y')
        
        # Technical indicators
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data['Returns'] = data['Close'].pct_change() * 100
        
        return data
    except:
        return pd.DataFrame()


def detect_significant_moves(df: pd.DataFrame) -> List[Dict]:
    """Detect significant price movements"""
    if df.empty:
        return []
    
    df['Price_Change_5d'] = df['Close'].pct_change(periods=5) * 100
    
    moves = []
    
    # Uptrends
    ups = df[df['Price_Change_5d'] > 5].tail(5)
    for date, row in ups.iterrows():
        moves.append({
            'type': 'UP',
            'date': date.strftime('%Y-%m-%d'),
            'change': row['Price_Change_5d'],
            'price': row['Close']
        })
    
    # Downtrends
    downs = df[df['Price_Change_5d'] < -5].tail(5)
    for date, row in downs.iterrows():
        moves.append({
            'type': 'DOWN',
            'date': date.strftime('%Y-%m-%d'),
            'change': row['Price_Change_5d'],
            'price': row['Close']
        })
    
    return sorted(moves, key=lambda x: x['date'], reverse=True)


def create_chart(ticker: str, data: pd.DataFrame) -> go.Figure:
    """Create stock chart"""
    if data.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker} Price', 'Volume'),
        vertical_spacing=0.1
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving averages
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20',
                   line=dict(color='orange', width=1)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50',
                   line=dict(color='blue', width=1)),
        row=1, col=1
    )
    
    # Volume
    colors = ['red' if c < o else 'green' 
              for c, o in zip(data['Close'], data['Open'])]
    
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume',
               marker_color=colors),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{ticker} Stock Analysis',
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    
    return fig


def answer_query(query: str, history: list) -> Tuple[str, Optional[go.Figure]]:
    """Main query handler"""
    
    ticker = extract_ticker(query)
    
    if not ticker:
        return "Please mention a stock ticker (e.g., AAPL, MSFT) or company name (e.g., Apple, Microsoft).", None
    
    # Fetch data
    data = fetch_stock_data(ticker)
    
    if data.empty:
        return f"Could not fetch data for {ticker}. Please check the ticker symbol.", None
    
    # Get current stats
    latest_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    change_today = ((latest_price / prev_price) - 1) * 100
    
    change_1w = ((data['Close'].iloc[-1] / data['Close'].iloc[-5]) - 1) * 100
    change_1m = ((data['Close'].iloc[-1] / data['Close'].iloc[-20]) - 1) * 100
    
    high_52w = data['High'].max()
    low_52w = data['Low'].min()
    
    query_lower = query.lower()
    
    # Intent: Why did it drop/rise?
    if 'why' in query_lower or 'reason' in query_lower:
        moves = detect_significant_moves(data)
        
        if 'drop' in query_lower or 'fall' in query_lower or 'down' in query_lower:
            drops = [m for m in moves if m['type'] == 'DOWN']
            if drops:
                response = f"**Analysis for {ticker}:**\n\n"
                response += f"Current Price: ${latest_price:.2f}\n\n"
                response += "**Recent Significant Drops:**\n"
                for drop in drops[:3]:
                    response += f"- {drop['date']}: {drop['change']:.2f}% (Price: ${drop['price']:.2f})\n"
                response += f"\n1-Week Change: {change_1w:+.2f}%\n"
                response += f"1-Month Change: {change_1m:+.2f}%\n"
            else:
                response = f"{ticker} hasn't had significant drops (>5%) recently.\n"
                response += f"1-Week: {change_1w:+.2f}%, 1-Month: {change_1m:+.2f}%"
        else:
            response = f"**{ticker} Analysis:**\n\n"
            response += f"Current: ${latest_price:.2f} ({change_today:+.2f}% today)\n"
            response += f"1-Week: {change_1w:+.2f}%\n"
            response += f"1-Month: {change_1m:+.2f}%\n"
        
        chart = create_chart(ticker, data)
        return response, chart
    
    # Intent: When did it go up?
    elif 'when' in query_lower:
        moves = detect_significant_moves(data)
        
        if 'up' in query_lower or 'rise' in query_lower or 'increase' in query_lower:
            ups = [m for m in moves if m['type'] == 'UP']
            if ups:
                response = f"**{ticker} Uptrend Dates:**\n\n"
                for up in ups[:5]:
                    response += f"📈 {up['date']}: +{up['change']:.2f}% (Price: ${up['price']:.2f})\n"
            else:
                response = f"{ticker} hasn't had significant uptrends (>5%) recently."
        else:
            response = f"**{ticker} Recent Movements:**\n\n"
            for move in moves[:5]:
                emoji = "📈" if move['type'] == 'UP' else "📉"
                response += f"{emoji} {move['date']}: {move['change']:+.2f}%\n"
        
        chart = create_chart(ticker, data)
        return response, chart
    
    # Intent: Show chart
    elif any(word in query_lower for word in ['chart', 'plot', 'show', 'graph', 'visualize']):
        response = f"**{ticker} Chart:**\n\n"
        response += f"Current: ${latest_price:.2f}\n"
        response += f"52-Week Range: ${low_52w:.2f} - ${high_52w:.2f}\n"
        
        chart = create_chart(ticker, data)
        return response, chart
    
    # Default: General info
    else:
        response = f"**{ticker} Stock Information:**\n\n"
        response += f"💰 Current Price: ${latest_price:.2f}\n"
        response += f"📊 Change Today: {change_today:+.2f}%\n"
        response += f"📈 1-Week: {change_1w:+.2f}%\n"
        response += f"📉 1-Month: {change_1m:+.2f}%\n"
        response += f"🔝 52W High: ${high_52w:.2f}\n"
        response += f"🔻 52W Low: ${low_52w:.2f}\n"
        
        chart = create_chart(ticker, data)
        return response, chart


def get_live_data(ticker: str) -> Tuple[str, Optional[go.Figure]]:
    """Get live/intraday data"""
    if not ticker:
        return "Please enter a ticker.", None
    
    try:
        ticker = ticker.upper()
        stock = yf.Ticker(ticker)
        
        # Try to get today's data
        live = stock.history(period='1d', interval='1m')
        
        if live.empty:
            # Fallback to recent data
            live = stock.history(period='5d')
        
        if live.empty:
            return f"No data available for {ticker}", None
        
        latest = live.iloc[-1]
        first = live.iloc[0]
        
        change = ((latest['Close'] / first['Close']) - 1) * 100
        
        response = f"**Live Data: {ticker}**\n\n"
        response += f"💵 Current: ${latest['Close']:.2f}\n"
        response += f"📊 Change: {change:+.2f}%\n"
        response += f"📈 High: ${latest['High']:.2f}\n"
        response += f"📉 Low: ${latest['Low']:.2f}\n"
        response += f"📦 Volume: {latest['Volume']:,.0f}\n"
        response += f"🕐 Updated: {live.index[-1].strftime('%Y-%m-%d %H:%M')}\n"
        
        # Create mini chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=live.index,
            y=live['Close'],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=f'{ticker} Intraday',
            yaxis_title='Price ($)',
            height=400
        )
        
        return response, fig
        
    except Exception as e:
        return f"Error: {str(e)}", None


# Gradio Interface
def create_demo():
    with gr.Blocks(title="Stock RAG Chatbot - Demo") as demo:
        gr.Markdown("# 📈 Stock Market RAG Chatbot (Demo)")
        gr.Markdown("Simple demo - no API keys needed!")
        
        with gr.Tab("💬 Chat"):
            chatbot_output = gr.Textbox(label="Response", lines=10)
            chart_output = gr.Plot(label="Chart")
            
            msg_input = gr.Textbox(
                label="Ask a question",
                placeholder="Why did Apple drop? When did Microsoft go up? Show me Tesla chart"
            )
            
            submit_btn = gr.Button("Ask", variant="primary")
            
            gr.Markdown("""
            **Try these:**
            - Why did Apple stock drop?
            - When did Microsoft go up?
            - Show me a chart for Tesla
            - What's happening with NVDA?
            """)
            
            submit_btn.click(
                answer_query,
                inputs=[msg_input, gr.State([])],
                outputs=[chatbot_output, chart_output]
            )
        
        with gr.Tab("📡 Live Data"):
            live_ticker = gr.Textbox(label="Ticker", placeholder="AAPL")
            live_btn = gr.Button("Get Live Data", variant="primary")
            
            live_text = gr.Textbox(label="Live Info", lines=8)
            live_chart = gr.Plot(label="Live Chart")
            
            live_btn.click(
                get_live_data,
                inputs=[live_ticker],
                outputs=[live_text, live_chart]
            )
        
        gr.Markdown("""
        ## About
        - **Task 2**: RAG-based chatbot with intent recognition
        - **Task 3**: Real-time data integration
        - No API key needed for this demo
        - Full version supports LLM integration (Groq/Gemini)
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)
