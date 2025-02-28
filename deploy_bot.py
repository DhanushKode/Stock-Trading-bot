import threading
import time
from collections import deque

import plotly.graph_objects as go
import requests
import streamlit as st

# Streamlit UI
st.title("📈 AI Stock Trading Bot")

# User Inputs
alpaca_api_key = st.text_input("Enter Alpaca API Key", type="password")
alpaca_secret_key = st.text_input("Enter Alpaca Secret Key", type="password")
symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, NVDA)", "AAPL")
price_threshold = st.number_input("Price Change Threshold (%)", min_value=0.1, value=0.3)
trade_quantity = st.number_input("Trade Quantity (Shares)", min_value=1, value=1)

# Data Storage for Live Graph
price_data = deque(maxlen=50)  # Store last 50 prices
time_data = deque(maxlen=50)  # Store timestamps

# Function to fetch stock price from Alpaca
def get_stock_price(symbol, api_key, secret_key):
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key
    }
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        return round(float(data["quote"]["ap"]), 2)  # Round to 2 decimals
    except Exception as e:
        st.error(f"Failed to fetch stock price: {e}")
        return None

# Function to place an order on Alpaca
def place_order(symbol, qty, side, api_key, secret_key):
    url = "https://paper-api.alpaca.markets/v2/orders"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "Content-Type": "application/json"
    }
    order_data = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": "market",
        "time_in_force": "gtc"
    }
    try:
        response = requests.post(url, json=order_data, headers=headers)
        if response.status_code == 200:
            st.success(f"✅ Order placed: {side.upper()} {qty} shares of {symbol}")
        else:
            st.error(f"❌ Order failed: {response.json()}")
    except Exception as e:
        st.error(f"Error placing order: {e}")

# Function to start trading bot in a thread
def trading_bot():
    if not alpaca_api_key or not alpaca_secret_key:
        st.error("⚠ Please enter your Alpaca API credentials!")
        return

    st.write("📊 Trading Bot Started...")
    prev_price = get_stock_price(symbol, alpaca_api_key, alpaca_secret_key)
    
    if prev_price:
        st.write(f"📊 Initial Price: {prev_price}")

        while True:
            latest_price = get_stock_price(symbol, alpaca_api_key, alpaca_secret_key)
            
            if latest_price:
                price_change = ((latest_price - prev_price) / prev_price) * 100
                price_change = round(price_change, 2)

                # Update graph data
                time_data.append(time.strftime("%H:%M:%S"))
                price_data.append(latest_price)

                if price_change > price_threshold:
                    st.success(f"📈 Price Increased by {price_change:.2f}% - BUY {trade_quantity} Shares!")
                    place_order(symbol, trade_quantity, "buy", alpaca_api_key, alpaca_secret_key)
                elif price_change < -price_threshold:
                    st.warning(f"📉 Price Dropped by {price_change:.2f}% - SELL {trade_quantity} Shares!")
                    place_order(symbol, trade_quantity, "sell", alpaca_api_key, alpaca_secret_key)
                else:
                    st.info(f"🔍 No significant change ({price_change:.2f}%) - Holding position.")

                prev_price = latest_price

            time.sleep(30)  # Check every 30 seconds

# Visualization Function
def live_chart():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(time_data), y=list(price_data), mode="lines+markers", name="Stock Price"))
    fig.update_layout(title=f"{symbol} Stock Price Live", xaxis_title="Time", yaxis_title="Price", template="plotly_dark")
    return fig

# Start button
if st.button("🚀 Start Trading Bot"):
    thread = threading.Thread(target=trading_bot, daemon=True)
    thread.start()

# Live Visualization
st.write("📊 **Live Stock Price Graph**")
chart_placeholder = st.empty()
while True:
    with chart_placeholder.container():
        st.plotly_chart(live_chart(), use_container_width=True)
    time.sleep(5)  # Update graph every 5 seconds
