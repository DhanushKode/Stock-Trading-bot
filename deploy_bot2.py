import os
import queue
import threading
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Streamlit UI
st.title("📈 AI - Stock Trading Bot")

# User Inputs
alpaca_api_key = st.text_input("Alpaca API Key", type="password")
alpaca_secret_key = st.text_input("Alpaca Secret Key", type="password")
tiingo_api_key = st.text_input("Tiingo API Key", type="password")
stock_symbol = st.text_input("Stock Symbol (e.g., NVDA)", "NVDA")
window_size = st.number_input("Prediction Window Size (days)", min_value=5, value=10)
trade_qty = st.number_input("Trade Quantity (shares)", min_value=1, value=10)

# Status Placeholder
status_placeholder = st.empty()

# Initialize Session State
if "bot_status" not in st.session_state:
    st.session_state.bot_status = "🔴 Bot Not Started"
if "status_queue" not in st.session_state:
    st.session_state.status_queue = queue.Queue()
if "bot_event" not in st.session_state:
    st.session_state.bot_event = threading.Event()

# Fetch Tiingo Data
def fetch_tiingo_data(symbol, api_key, start_date, end_date):
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate={start_date}&endDate={end_date}&token={api_key}"
    response = requests.get(url)
    data = response.json()
    if not data:
        raise ValueError(f"No data returned for {symbol}")
    return data

# Load Dataset
def dataset_loader(stock_name, api_key, status_queue):
    try:
        start_date = "2012-01-01"
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        raw_data = fetch_tiingo_data(stock_name, api_key, start_date, end_date)
        data = pd.DataFrame(raw_data)
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')
        data = data.sort_index()
        return data['close']
    except Exception as e:
        status_queue.put(f"❌ Failed to fetch data: {e}")
        return pd.Series(100 + np.cumsum(np.random.normal(0, 1, 100)))

# Fetch Current Price
def get_current_price(symbol, api_key, secret_key, status_queue):
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest"
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        return float(data["trade"]["p"])
    except Exception as e:
        status_queue.put(f"❌ Failed to fetch price: {e}")
        return None

# Place Order
def place_order(trading_client, symbol, qty, side, limit_price, status_queue):
    if limit_price <= 0:
        status_queue.put(f"❌ Invalid limit price: {limit_price}")
        return
    try:
        order = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=round(limit_price, 2)
        )
        response = trading_client.submit_order(order)
        status_queue.put(f"✅ Order placed: {side} {qty} shares of {symbol} at ${round(limit_price, 2)}")
    except Exception as e:
        status_queue.put(f"❌ Error placing {side} order: {e}")

# ML Trader
class ML_Trader:
    def __init__(self, window_size):
        self.window_size = window_size
        self.model = LinearRegression()
        self.inventory = []

    def train(self, data):
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data.iloc[i:i + self.window_size].values)
            y.append(data.iloc[i + self.window_size])
        X, y = np.array(X), np.array(y)
        if X.shape[0] > 0:
            self.model.fit(X, y)

    def predict(self, recent_data):
        return self.model.predict(np.array(recent_data).reshape(1, -1))[0]

# Trading Bot Logic
def trading_bot(status_queue, stop_event, api_key, secret_key, tiingo_key, symbol, window_size, qty):
    if not api_key or not secret_key or not tiingo_key:
        status_queue.put("⚠ Please enter all API keys!")
        return

    trading_client = TradingClient(api_key, secret_key, paper=True)
    data = dataset_loader(symbol, tiingo_key, status_queue)
    trader = ML_Trader(window_size)
    train_data, _ = train_test_split(data, test_size=0.2, shuffle=False)
    trader.train(train_data)

    status_queue.put("🚀 Trading Bot Started...")
    status_queue.put(f"📊 Loaded {len(data)} data points")

    while stop_event.is_set():
        try:
            current_price = get_current_price(symbol, api_key, secret_key, status_queue)
            if current_price is None:
                time.sleep(60)
                continue

            data = pd.concat([data, pd.Series([current_price], index=[pd.Timestamp.now()])])
            recent_data = data.iloc[-window_size:].values.flatten()
            predicted_price = trader.predict(recent_data).item()

            status_queue.put(f"📈 Current Price: ${current_price:.2f}, Predicted: ${predicted_price:.2f}")

            if predicted_price > current_price and len(trader.inventory) == 0:
                trader.inventory.append(current_price)
                status_queue.put(f"💰 AI Trader bought: ${current_price:.2f}")
                place_order(trading_client, symbol, qty, OrderSide.BUY, current_price, status_queue)

            elif len(trader.inventory) > 0 and (predicted_price < current_price or
                                                current_price < trader.inventory[0] * 0.98):
                time.sleep(10)
                buy_price = trader.inventory.pop(0)
                sell_price = current_price * 1.01  # 1% above current price
                profit = round(current_price - buy_price, 2)
                status_queue.put(f"💰 AI Trader sold: ${current_price:.2f}, Profit: ${profit:.2f}")
                place_order(trading_client, symbol, qty, OrderSide.SELL, sell_price, status_queue)

            if len(data) > window_size * 2:
                trader.train(data[-window_size * 2:])

            time.sleep(60)

        except Exception as e:
            status_queue.put(f"❌ Error in trading loop: {e}")
            time.sleep(60)

# Start Button
if st.button("🚀 Start Trading Bot"):
    if not st.session_state.bot_event.is_set():
        st.session_state.bot_event.set()
        thread = threading.Thread(
            target=trading_bot,
            args=(st.session_state.status_queue, st.session_state.bot_event, alpaca_api_key, alpaca_secret_key, tiingo_api_key, stock_symbol, window_size, trade_qty),
            daemon=True
        )
        thread.start()

# Stop Button
if st.button("🛑 Stop Trading Bot"):
    st.session_state.bot_event.clear()
    st.session_state.bot_status = "🔴 Bot Stopped"

# Update UI from Queue
if st.session_state.bot_event.is_set():
    with status_placeholder.container():
        if not st.session_state.status_queue.empty():
            st.session_state.bot_status = st.session_state.status_queue.get()
        st.info(st.session_state.bot_status)
else:
    status_placeholder.info(st.session_state.bot_status)
