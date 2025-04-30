import base64
import logging
import os
import queue
import socket
import threading
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Set up logging to debug issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to get Base64 string from MP4 (with caching)
@st.cache_data
def get_base64_of_file(file_path):
    try:
        with open(file_path, "rb") as video_file:
            encoded_string = base64.b64encode(video_file.read()).decode("utf-8")
        logger.info("Successfully encoded MP4 to Base64")
        return encoded_string
    except Exception as e:
        logger.error(f"Failed to encode MP4: {e}")
        raise

# Fetch Tiingo Data
def fetch_tiingo_data(symbol, api_key, start_date, end_date):
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    headers = {"Authorization": f"Token {api_key}"}
    params = {"startDate": start_date, "endDate": end_date}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data:
            raise ValueError(f"No data returned for {symbol}")
        logger.info(f"Fetched {len(data)} data points for {symbol} from Tiingo")
        return data
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            logger.error(f"403 Forbidden: Check API key or usage limits for {symbol}. Error: {e}")
            st.error(f"403 Forbidden: Verify API key or reduce date range for {symbol}. Contact Tiingo support if needed.")
        else:
            logger.error(f"Failed to fetch Tiingo data: {e}")
            st.error(f"Data fetch failed: {e}")
        return []

# Load Dataset
def dataset_loader(stock_name, api_key, status_queue):
    try:
        start_date = "2024-01-01"
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        raw_data = fetch_tiingo_data(stock_name, api_key, start_date, end_date)
        data = pd.DataFrame(raw_data)
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')
        data = data.sort_index()
        logger.info(f"Loaded dataset for {stock_name} with {len(data)} entries")
        return data['close']
    except Exception as e:
        status_queue.put(f"❌ Failed to fetch data: {e}")
        logger.error(f"Dataset loading failed: {e}")
        return pd.Series(100 + np.cumsum(np.random.normal(0, 1, 100)))

# Fetch Current Price - Enhanced with DNS check and stronger fallback
def get_current_price(symbol, api_key, secret_key, status_queue, last_price=None):
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest"
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
    max_retries = 5
    last_successful_price = last_price

    # Preliminary DNS check
    try:
        socket.getaddrinfo("data.alpaca.markets", 443)
    except socket.gaierror as e:
        logger.error(f"DNS resolution failed for data.alpaca.markets: {e}")
        status_queue.put(f"⚠ DNS resolution failed: Check your network or contact Alpaca support.")
        if last_successful_price is not None:
            simulated_price = last_successful_price + np.random.normal(0, 0.5)
            simulated_price = max(1, simulated_price)
            return simulated_price, last_successful_price
        return None, last_successful_price

    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching current price for {symbol} from Alpaca (Attempt {attempt + 1}/{max_retries})")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if "trade" not in data or "p" not in data["trade"]:
                raise ValueError(f"Invalid response from Alpaca: {data}")
            price = float(data["trade"]["p"])
            last_successful_price = price
            logger.info(f"Fetched current price for {symbol}: ${price}")
            return price, last_successful_price
        except Exception as e:
            logger.error(f"Price fetch failed (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                if last_successful_price is not None:
                    simulated_price = last_successful_price + np.random.normal(0, 0.5)
                    simulated_price = max(1, simulated_price)
                    status_queue.put(f"⚠ Failed to fetch price after {max_retries} attempts, using simulated price: ${simulated_price:.2f}")
                    logger.warning(f"Using simulated price: ${simulated_price:.2f}")
                    return simulated_price, last_successful_price
                status_queue.put(f"❌ Failed to fetch price after {max_retries} attempts: {e}")
                return None, last_successful_price
            time.sleep(2 ** attempt)
    return None, last_successful_price

# Place Order - Optimized for execution
def place_order(trading_client, symbol, qty, side, limit_price, status_queue):
    if limit_price <= 0:
        status_queue.put(f"❌ Invalid limit price: {limit_price}")
        logger.warning(f"Invalid limit price: {limit_price}")
        return False
    try:
        if side == OrderSide.BUY:
            limit_price = round(limit_price * 0.998, 2)
        else:
            limit_price = round(limit_price * 1.002, 2)
        order = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price
        )
        response = trading_client.submit_order(order)
        status_queue.put(f"✅ Order placed: {side} {qty} shares of {symbol} at ${limit_price:.2f}")
        logger.info(f"Order placed: {side} {qty} shares of {symbol} at ${limit_price:.2f}")
        return True
    except Exception as e:
        status_queue.put(f"❌ Error placing {side} order: {e}")
        logger.error(f"Order placement failed: {e}")
        return False

# ML Trader - Added noise to predictions for demo
class ML_Trader:
    def __init__(self, window_size):
        self.window_size = window_size
        self.model = LinearRegression()
        self.inventory = []
        self.iterations_since_last_sell = 0  # Track iterations since last sell
        logger.info(f"Initialized ML_Trader with window_size={window_size}")

    def train(self, data):
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data.iloc[i:i + self.window_size].values)
            y.append(data.iloc[i + self.window_size])
        X, y = np.array(X), np.array(y)
        if X.shape[0] > 0:
            self.model.fit(X, y)
            logger.info(f"Trained model with {X.shape[0]} samples")
        else:
            logger.warning("No data to train the model")

    def predict(self, recent_data):
        prediction = self.model.predict(np.array(recent_data).reshape(1, -1))[0]
        # Add noise for demo to ensure trades
        prediction += np.random.normal(0, 0.1)  # Small noise (±0.1)
        logger.info(f"Predicted price: ${prediction:.2f}")
        return prediction

# Trading Bot Logic - Fixed typo in trader.inventory
def trading_bot(status_queue, stop_event, api_key, secret_key, tiingo_key, symbol, window_size, qty, max_exposure=5000):
    logger.info("Starting trading bot...")
    if not api_key or not secret_key or not tiingo_key:
        status_queue.put("⚠ Please enter all API keys!")
        logger.warning("Missing API keys")
        return

    trading_client = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            trading_client = TradingClient(api_key, secret_key, paper=True)
            account = trading_client.get_account()
            status_queue.put(f"✅ Connected to Alpaca. Buying Power: ${account.buying_power}")
            logger.info("Initialized Alpaca TradingClient in paper mode")
            break
        except Exception as e:
            logger.error(f"Trading client initialization failed (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                status_queue.put(f"❌ Failed to connect to Alpaca after {max_retries} attempts: {e}. Switching to simulation mode.")
                logger.warning("Falling back to simulation mode")
                trading_client = None
            time.sleep(2 ** attempt)

    data = dataset_loader(symbol, tiingo_key, status_queue)
    trader = ML_Trader(window_size)
    train_data, _ = train_test_split(data, test_size=0.2, shuffle=False)
    trader.train(train_data)

    status_queue.put("🚀 Trading Bot Started...")
    status_queue.put(f"📊 Loaded {len(data)} data points")
    logger.info("Trading bot started")
    total_profit = 0.0
    trade_count = 0
    last_price = None

    if data.empty:
        current_price = 100.0
        last_price = 100.0
    else:
        current_price = data.iloc[-1]
        last_price = current_price

    while not stop_event.is_set():
        try:
            logger.info("Entering trading loop iteration")
            current_price, last_price = get_current_price(symbol, api_key, secret_key, status_queue, last_price)
            if current_price is None:
                current_price = last_price if last_price is not None else 100.0 + np.random.normal(0, 1)
                logger.warning(f"Using fallback price: ${current_price:.2f}")

            data = pd.concat([data, pd.Series([current_price], index=[pd.Timestamp.now()])])
            recent_data = data.iloc[-window_size:].values.flatten()
            predicted_price = trader.predict(recent_data)

            buy_threshold = current_price * 1.00002  # Reduced to 0.002%
            sell_profit_threshold = trader.inventory[0] * 1.001 if trader.inventory else current_price * 1.001  # 0.1% profit
            sell_stop_loss = trader.inventory[0] * 0.999 if trader.inventory else current_price * 0.999  # 0.1% stop-loss
            buy_signal = predicted_price >= buy_threshold and len(trader.inventory) == 0
            sell_signal = len(trader.inventory) > 0 and (current_price >= sell_profit_threshold or current_price <= sell_stop_loss)
            current_exposure = sum(trader.inventory) * qty if trader.inventory else 0

            # Track iterations since last sell to force a sell if inventory is stuck
            if len(trader.inventory) > 0:
                trader.iterations_since_last_sell += 1
                if trader.iterations_since_last_sell >= 5:  # Force sell after 5 iterations
                    sell_signal = True
                    logger.info("Forcing sell due to inventory stuck for 5 iterations")
            else:
                trader.iterations_since_last_sell = 0

            logger.info(f"Buy Signal Evaluation: Predicted=${predicted_price:.2f} >= Threshold=${buy_threshold:.2f} -> {buy_signal}")
            logger.debug(f"Current: ${current_price:.2f}, Predicted: ${predicted_price:.2f}, Buy Threshold: ${buy_threshold:.2f}, Sell Profit: ${sell_profit_threshold:.2f}, Sell Stop: ${sell_stop_loss:.2f}, Buy Signal: {buy_signal}, Sell Signal: {sell_signal}, Inventory: {len(trader.inventory)}")
            status_queue.put(f"📈 {symbol}: ${current_price:.2f} | Predicted: ${predicted_price:.2f} | Inventory: {len(trader.inventory)}")
            status_queue.put(f"📊 PriceHistoryUpdate: {pd.Timestamp.now()} | {current_price}")

            # Debug why buy condition might fail
            exposure_check = current_exposure + (current_price * qty) <= max_exposure
            logger.info(f"Exposure Check: Current Exposure=${current_exposure:.2f}, New Exposure=${current_price * qty:.2f}, Max Exposure=${max_exposure:.2f} -> {exposure_check}")

            if buy_signal and exposure_check:
                if trading_client and place_order(trading_client, symbol, qty, OrderSide.BUY, current_price, status_queue):
                    trader.inventory.append(current_price)
                    trade_count += 1
                    status_queue.put(f"💰 BUY EXECUTED: {trade_count} trades | Price: ${current_price:.2f}")
                    status_queue.put(f"💰 TradeHistoryUpdate: {pd.Timestamp.now()} | BUY | {current_price} | {qty} | 0")
                    logger.info(f"Buy executed at ${current_price:.2f}")
                else:
                    trader.inventory.append(current_price)
                    trade_count += 1
                    status_queue.put(f"💰 SIMULATED BUY: {trade_count} trades | Price: ${current_price:.2f}")
                    status_queue.put(f"💰 TradeHistoryUpdate: {pd.Timestamp.now()} | SIMULATED BUY | {current_price} | {qty} | 0")
                    logger.info(f"Simulated buy at ${current_price:.2f}. Trading client available: {trading_client is not None}")
            else:
                logger.info(f"Buy not executed: Buy Signal={buy_signal}, Exposure Check={exposure_check}")

            if sell_signal:
                if len(trader.inventory) > 0:
                    buy_price = trader.inventory.pop(0)
                    profit = (current_price - buy_price) * qty
                    total_profit += profit
                    trader.iterations_since_last_sell = 0  # Reset counter
                    if trading_client and place_order(trading_client, symbol, qty, OrderSide.SELL, current_price, status_queue):
                        trade_count += 1
                        status_queue.put(f"💰 SELL EXECUTED: {trade_count} trades | Profit: ${profit:.2f} | Total: ${total_profit:.2f}")
                        status_queue.put(f"💰 TradeHistoryUpdate: {pd.Timestamp.now()} | SELL | {current_price} | {qty} | {profit}")
                        logger.info(f"Sell executed at ${current_price:.2f} with profit ${profit:.2f}")
                    else:
                        trade_count += 1
                        status_queue.put(f"💰 SIMULATED SELL: {trade_count} trades | Profit: ${profit:.2f} | Total: ${total_profit:.2f}")
                        status_queue.put(f"💰 TradeHistoryUpdate: {pd.Timestamp.now()} | SIMULATED SELL | {current_price} | {qty} | {profit}")
                        logger.info(f"Simulated sell at ${current_price:.2f} with profit ${profit:.2f}")
                    logger.info(f"Inventory after sell: {len(trader.inventory)}")

            if len(data) % 20 == 0:
                trader.train(data[-window_size*2:])
                logger.info("Retrained model with new data")

            time.sleep(10)

        except Exception as e:
            status_queue.put(f"❌ Error in trading loop: {e}")
            logger.error(f"Trading loop error: {e}")
            time.sleep(60)

# Introductory Page
def intro_page():
    st.markdown(common_css, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        st.title("Welcome to Your AI Stock Trading Journey!")

        st.markdown("""
            ### Discover the Power of Automated Trading
            Welcome to our AI Stock Trading Bot, designed to simplify your entry into the world of stock trading! Whether you're a beginner or an experienced investor, this tool helps you buy and sell stocks like NVIDIA (NVDA) with ease, using smart predictions and real-time data. Our bot operates in paper trading mode, allowing you to practice without financial risk.
        """)

        st.markdown("""
            ### How It Works
            - **Smart Predictions**: Powered by machine learning, the bot analyzes historical data to predict price movements.
            - **Easy Setup**: Choose your stock (e.g., NVDA, AAPL, MSFT) and set your trading preferences.
            - **24/7 Automation**: Runs continuously, executing trades based on small price changes.
            - **Risk Management**: Limits exposure to protect your virtual portfolio.
        """)

        st.markdown("""
            ### Get Started
            Click below to proceed to the trading bot and start your journey with personalized settings and real-time trading!
        """)

        if st.button("Start Trading Now", key="proceed_button"):
            st.session_state.page = "trading_bot"
            st.session_state.selected_stock = "NVDA"
            st.session_state.window_size = 10
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# Trading Bot Page
def trading_bot_page():
    st.markdown(common_css, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        st.title("📈 AI Stock Trading Bot")

        # User Inputs
        with st.container():
            st.subheader("🔑 API Keys")
            alpaca_api_key = st.text_input("Alpaca API Key", type="password", key="alpaca_key")
            alpaca_secret_key = st.text_input("Alpaca Secret Key", type="password", key="alpaca_secret")
            tiingo_api_key = st.text_input("Tiingo API Key", type="password", key="tiingo_key")

        with st.container():
            st.subheader("⚙ Trading Parameters")
            stock_symbol = st.text_input("Stock Symbol (e.g., AAPL)", st.session_state.get("selected_stock", "NVDA"), key="stock_symbol")
            window_size = st.number_input("Prediction Window Size (days)", min_value=5, value=st.session_state.get("window_size", 10))
            trade_qty = st.number_input("Trade Quantity (shares)", min_value=1, value=2)
            max_exposure = st.number_input("Max Exposure ($)", min_value=100, value=5000, step=100)

        # Status Dashboard
        st.subheader("📊 Bot Status Dashboard")
        if "current_price" not in st.session_state:
            st.session_state.current_price = "N/A"
        if "last_action" not in st.session_state:
            st.session_state.last_action = "None"
        if "bot_status" not in st.session_state:
            st.session_state.bot_status = "🔴 Bot Not Started"
        if "total_profit" not in st.session_state:
            st.session_state.total_profit = 0.0
        if "trade_count" not in st.session_state:
            st.session_state.trade_count = 0
        if "price_history" not in st.session_state:
            st.session_state.price_history = []
        if "trade_history" not in st.session_state:
            st.session_state.trade_history = []
        if "status_log" not in st.session_state:
            st.session_state.status_log = []

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Current Price", st.session_state.current_price)
        with col2:
            st.metric("Last Action", st.session_state.last_action)
        with col3:
            st.metric("Bot Status", st.session_state.bot_status)
        with col4:
            st.metric("Total Profit", f"${st.session_state.total_profit:.2f}")
        with col5:
            st.metric("Trade Count", st.session_state.trade_count)

        # Status Log with Scrollable Box and Refresh Button
        st.subheader("📜 Status Log")
        status_placeholder = st.empty()

        # Refresh Button to Update Status Log
        if st.button("🔄 Refresh Status Log"):
            logger.info("Refresh Status Log button clicked")
            while not st.session_state.status_queue.empty():
                status = st.session_state.status_queue.get()
                st.session_state.status_log.append(status)
                if len(st.session_state.status_log) > 10:  # Keep only the last 10 messages
                    st.session_state.status_log.pop(0)
                if status.startswith("📈"):
                    parts = status.split("|")
                    if len(parts) > 0:
                        price_part = parts[0].split("$")[1]
                        st.session_state.current_price = f"${float(price_part):.2f}"
                elif status.startswith("📊 PriceHistoryUpdate:"):
                    try:
                        parts = status.split("|")
                        if len(parts) >= 3:
                            timestamp = parts[1].strip()
                            price = float(parts[2].strip())
                            st.session_state.price_history.append((pd.Timestamp(timestamp), price))
                    except (ValueError, IndexError) as e:
                        logger.error(f"Failed to parse PriceHistoryUpdate: {status}, Error: {e}")
                        continue
                elif status.startswith("💰"):
                    st.session_state.last_action = status
                    if "Total Profit" in status:
                        profit_part = status.split("Total Profit: $")[1]
                        st.session_state.total_profit = float(profit_part.split("}")[0]) if profit_part and profit_part.split("}")[0] else st.session_state.total_profit
                    if "trades" in status:
                        trade_part = status.split("trades |")[0].split(" ")[-1]
                        st.session_state.trade_count = int(trade_part) if trade_part and trade_part.isdigit() else st.session_state.trade_count
                elif status.startswith("💰 TradeHistoryUpdate:"):
                    try:
                        parts = status.split("|")
                        if len(parts) >= 6:
                            timestamp = pd.Timestamp(parts[1].strip())
                            action = parts[2].strip()
                            price = float(parts[3].strip())
                            qty = int(parts[4].strip())
                            profit = float(parts[5].strip())
                            st.session_state.trade_history.append((timestamp, action, price, qty, profit))
                    except (ValueError, IndexError) as e:
                        logger.error(f"Failed to parse TradeHistoryUpdate: {status}, Error: {e}")
                        continue
            status_log_html = "<div class='status-box'>"
            for status in st.session_state.status_log:
                status_log_html += f"{status}<br>"
            status_log_html += "</div>"
            status_placeholder.markdown(status_log_html, unsafe_allow_html=True)

        # Trade History
        st.subheader("📜 Trade History")
        if st.session_state.trade_history:
            trade_df = pd.DataFrame(
                st.session_state.trade_history,
                columns=["Timestamp", "Action", "Price", "Quantity", "Profit"]
            )
            st.dataframe(trade_df)
        else:
            st.write("No trades yet.")

        # Price Chart
        st.subheader("📉 Price Trend")
        if st.session_state.price_history:
            price_df = pd.DataFrame(st.session_state.price_history, columns=["Timestamp", "Price"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=price_df["Timestamp"], y=price_df["Price"], mode="lines+markers", name="Price"))
            
            if st.session_state.trade_history:
                trade_df = pd.DataFrame(st.session_state.trade_history, columns=["Timestamp", "Action", "Price", "Quantity", "Profit"])
                buy_trades = trade_df[trade_df["Action"].str.contains("BUY")]
                sell_trades = trade_df[trade_df["Action"].str.contains("SELL")]
                fig.add_trace(go.Scatter(
                    x=buy_trades["Timestamp"], y=buy_trades["Price"],
                    mode="markers", marker=dict(symbol="triangle-up", size=10, color="green"),
                    name="Buys"
                ))
                fig.add_trace(go.Scatter(
                    x=sell_trades["Timestamp"], y=sell_trades["Price"],
                    mode="markers", marker=dict(symbol="triangle-down", size=10, color="red"),
                    name="Sells"
                ))
            
            fig.update_layout(
                title=f"{stock_symbol} Price Trend",
                xaxis_title="Time",
                yaxis_title="Price ($)",
                height=500  # Increased chart height
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No price data yet.")

        # Initialize Session State
        if "status_queue" not in st.session_state:
            st.session_state.status_queue = queue.Queue()
        if "bot_event" not in st.session_state:
            st.session_state.bot_event = threading.Event()
        if "bot_thread" not in st.session_state:
            st.session_state.bot_thread = None
        if "is_bot_running" not in st.session_state:
            st.session_state.is_bot_running = False

        # Manual Trade Button
        if st.button("📋 Manual Buy", disabled=st.session_state.is_bot_running):
            if "current_price" in st.session_state and st.session_state.current_price != "N/A":
                current_price = float(st.session_state.current_price.replace("$", ""))
                if place_order(TradingClient(alpaca_api_key, alpaca_secret_key, paper=True), stock_symbol, trade_qty, OrderSide.BUY, current_price, st.session_state.status_queue):
                    st.session_state.trade_count += 1
                    st.session_state.status_queue.put(f"💰 MANUAL BUY: {st.session_state.trade_count} trades | Price: ${current_price:.2f}")
                    st.session_state.trade_history.append((pd.Timestamp.now(), "MANUAL BUY", current_price, trade_qty, 0))

        # Start and Stop Buttons with Persistent Availability
        col_start_stop = st.columns([1, 1])
        with col_start_stop[0]:
            if not st.session_state.is_bot_running and st.button("🚀 Start Trading Bot"):
                if not st.session_state.bot_event.is_set() and all([alpaca_api_key, alpaca_secret_key, tiingo_api_key]):
                    logger.info("Start Trading Bot button clicked")
                    st.session_state.bot_event = threading.Event()
                    st.session_state.status_queue.queue.clear()
                    st.session_state.bot_thread = threading.Thread(
                        target=trading_bot,
                        args=(st.session_state.status_queue, st.session_state.bot_event, 
                              alpaca_api_key, alpaca_secret_key, tiingo_api_key, 
                              stock_symbol, window_size, trade_qty, max_exposure),
                        daemon=True
                    )
                    st.session_state.bot_thread.start()
                    st.session_state.is_bot_running = True
                    st.session_state.bot_status = "🟢 Bot Running"
                    status_placeholder.markdown('<div class="status-box">🚀 Starting trading bot...</div>', unsafe_allow_html=True)
        with col_start_stop[1]:
            if st.button("🛑 Stop Trading Bot"):
                logger.info("Stop Trading Bot button clicked")
                st.session_state.bot_event.set()
                if st.session_state.bot_thread:
                    st.session_state.bot_thread.join(timeout=5)
                st.session_state.is_bot_running = False
                st.session_state.bot_status = "🔴 Bot Stopped"
                status_placeholder.markdown('<div class="status-box">🛑 Trading bot stopped</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# Load the MP4 file (cached)
try:
    mp4_base64 = get_base64_of_file(r"C:\Users\HP\Downloads\BOT PROJECT\stock-market-price-chart.mp4")
except Exception as e:
    st.error(f"Failed to load MP4 file: {e}")
    mp4_base64 = ""

# Custom CSS and HTML for Video Background and Adjusted Layout
common_css = f"""
    <style>
    html, body, [data-testid="stAppViewContainer"] > .stApp {{
        margin: 0;
        padding: 0;
        overflow: hidden;
        background: #f4f6f9;  /* Fallback color if video fails */
    }}
    .video-background {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        z-index: -1;
        opacity: 0.7;
    }}
    [data-testid="stAppViewContainer"] {{
        position: relative;
        z-index: 1;
    }}
    .content-container {{
        max-width: 800px;  /* Increased width for better layout */
        margin: 0 auto;
        padding: 30px;
    }}
    h1 {{
        color: #FFD700 !important;  /* Gold color for title */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        font-family: 'Arial', sans-serif;
    }}
    h2, h3, p, label, div {{
        color: #ffffff !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
        font-family: 'Arial', sans-serif;
    }}
    .status-box {{
        background: rgba(0, 0, 0, 0.7);
        color: #ffffff !important;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        border: 1px solid #e0e0e0;
        max-height: 200px;  /* Scrollable status box */
        overflow-y: auto;
    }}
    .stTextInput > div > input {{
        background-color: rgba(255, 255, 255, 0.9);
        color: #000000 !important;
        border: 1px solid #cccccc;
        border-radius: 5px;
        padding: 8px;
    }}
    .stNumberInput > div > input {{
        background-color: rgba(255, 255, 255, 0.9);
        color: #000000 !important;
        border: 1px solid #cccccc;
        border-radius: 5px;
        padding: 8px;
    }}
    .stButton > button:nth-child(1) {{
        background-color: #2196F3;
        color: #ffffff !important;
        border-radius: 5px;
        padding: 10px 20px;
    }}
    .stButton > button:nth-child(1):hover {{
        background-color: #1976D2;
    }}
    .stButton > button:nth-child(2) {{
        background-color: #4CAF50;
        color: #ffffff !important;
        border-radius: 5px;
        padding: 10px 20px;
    }}
    .stButton > button:nth-child(2):hover {{
        background-color: #45a049;
    }}
    .stButton > button:nth-child(3) {{
        background-color: #ff0000;
        color: #ffffff !important;
        border-radius: 5px;
        padding: 10px 20px;
    }}
    .stButton > button:nth-child(3):hover {{
        background-color: #cc0000;
    }}
    </style>
    <video autoplay muted loop class="video-background">
        <source src="data:video/mp4;base64,{mp4_base64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
"""

# Inject JavaScript to Fix Autocomplete Attributes
components.html("""
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        const passwordInputs = document.querySelectorAll('input[type="password"]');
        passwordInputs.forEach(input => {
            input.setAttribute('autocomplete', 'new-password');
        });
        const symbolInput = document.querySelector('input[placeholder="Stock Symbol (e.g., AAPL)"]');
        if (symbolInput) {
            symbolInput.setAttribute('autocomplete', 'off');
        }
    });
    </script>
""", height=0)

# Main App Logic
if "page" not in st.session_state:
    st.session_state.page = "intro"

if st.session_state.page == "intro":
    intro_page()
elif st.session_state.page == "trading_bot":
    trading_bot_page()