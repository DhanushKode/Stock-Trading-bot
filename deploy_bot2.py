import queue
import random
import threading
import time

import requests
import streamlit as st

st.title("📈 AI Stock Trading Bot")

# User Inputs
alpaca_api_key = st.text_input("Alpaca API Key", type="password")
alpaca_secret_key = st.text_input("Alpaca Secret Key", type="password")
symbol = st.text_input("Stock Symbol", "TSLA").upper()
price_threshold = st.number_input("Price Change Threshold (%)", min_value=0.01, value=0.5)
trade_quantity = st.number_input("Trade Quantity (Shares)", min_value=1, value=1, step=1)
simulate_prices = st.checkbox("Simulate Price Changes", value=True)
max_shares = st.number_input("Max Shares to Hold", min_value=1, value=10, step=1)

# Initialize session state for UI
if "logs" not in st.session_state:
    st.session_state.logs = []
if "bot_active" not in st.session_state:
    st.session_state.bot_active = False

# Thread-safe queue for logs
log_queue = queue.Queue()

# Thread-safe control
bot_running_event = threading.Event()

# Function to fetch stock price
def get_stock_price(symbol, api_key, secret_key, simulate=False):
    base_price = 262.67  # TSLA, March 9, 02:30 AM PST
    if simulate:
        price = base_price * (1 + random.uniform(-0.05, 0.05))
        log_queue.put(f"Debug: Simulated price generated: {round(price, 2)}")
        return round(price, 2), None
    
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest"
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
    for attempt in range(3):  # Retry up to 3 times
        try:
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            price = round(float(response.json()["quote"]["ap"]), 2)
            log_queue.put(f"Debug: API price fetched: {price}")
            return price, None
        except Exception as e:
            log_queue.put(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(1)
    return None, f"Error fetching price after 3 attempts"

# Function to check current position
def get_position(symbol, api_key, secret_key):
    url = f"https://paper-api.alpaca.markets/v2/positions/{symbol}"
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            return float(response.json()["qty"]), None
        elif response.status_code == 404:
            return 0, None
        else:
            return None, f"Error checking position: {response.status_code} - {response.json()['message']}"
    except Exception as e:
        log_queue.put(f"Error checking position: {e}")
        return None, f"Error checking position: {e}"

# Function to place a bracket order (buy)
def place_bracket_order(symbol, qty, buy_price, api_key, secret_key):
    url = "https://paper-api.alpaca.markets/v2/orders"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "Content-Type": "application/json"
    }
    take_profit_price = round(buy_price * 1.015, 2)  # 1.5% profit
    stop_loss_price = round(buy_price * 0.95, 2)      # 5% loss
    if stop_loss_price > (buy_price - 0.05):
        stop_loss_price = round(buy_price - 0.05, 2)
    
    order_data = {
        "symbol": symbol,
        "qty": str(qty),
        "side": "buy",
        "type": "market",
        "time_in_force": "gtc",
        "order_class": "bracket",
        "take_profit": {"limit_price": str(take_profit_price)},
        "stop_loss": {"stop_price": str(stop_loss_price)}
    }
    try:
        log_queue.put(f"Debug: Sending bracket order - Buy: {buy_price}, TP: {take_profit_price}, SL: {stop_loss_price}")
        response = requests.post(url, json=order_data, headers=headers, timeout=5)
        if response.status_code == 200:
            order_id = response.json()["id"]
            return True, f"✅ Bracket BUY {qty} shares of {symbol} at {buy_price}, TP {take_profit_price}, SL {stop_loss_price} (Order ID: {order_id})"
        else:
            error_msg = response.json().get("message", "Unknown error")
            return False, f"❌ Bracket order failed: {response.status_code} - {error_msg}"
    except Exception as e:
        log_queue.put(f"Error placing bracket order: {e}")
        return False, f"Error placing bracket order: {e}"

# Function to place a simple market order (sell)
def place_order(symbol, qty, side, api_key, secret_key):
    url = "https://paper-api.alpaca.markets/v2/orders"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "Content-Type": "application/json"
    }
    order_data = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "market",
        "time_in_force": "gtc"
    }
    try:
        response = requests.post(url, json=order_data, headers=headers, timeout=5)
        if response.status_code == 200:
            return True, f"✅ {side.upper()} {qty} shares of {symbol} at market price"
        else:
            error_msg = response.json().get("message", "Unknown error")
            return False, f"❌ {side.upper()} failed: {response.status_code} - {error_msg}"
    except Exception as e:
        log_queue.put(f"Error placing {side} order: {e}")
        return False, f"Error placing {side} order: {e}"

# Trading bot logic
def trading_bot(api_key, secret_key, symbol, threshold, quantity, running_event, simulate, max_shares):
    log_queue.put("📊 Trading Bot Started Successfully!")
    price, error = get_stock_price(symbol, api_key, secret_key, simulate)
    prev_price = price if price is not None else 262.67
    last_trade_time = 0
    
    # Initial position sync
    api_position, pos_error = get_position(symbol, api_key, secret_key)
    local_position = api_position if api_position is not None else 0
    if pos_error:
        log_queue.put(pos_error)
    log_queue.put(f"📊 Initial Position: {local_position} shares, Initial Price: {prev_price}")
    
    iteration = 0
    while running_event.is_set():
        latest_price, error = get_stock_price(symbol, api_key, secret_key, simulate)
        if error:
            log_queue.put(error)
            time.sleep(1)
            continue
        
        price_change = round(((latest_price - prev_price) / prev_price) * 100, 2)
        log_queue.put(f"🔍 Price: {latest_price}, Change: {price_change}%")
        
        # Sync position with API every 10 iterations
        iteration += 1
        if iteration % 10 == 0:
            api_position, pos_error = get_position(symbol, api_key, secret_key)
            if pos_error:
                log_queue.put(pos_error)
            elif api_position != local_position:
                log_queue.put(f"⚠ Position sync: Local {local_position} -> API {api_position}")
                local_position = api_position

        current_time = time.time()
        time_since_last_trade = current_time - last_trade_time

        if price_change > threshold and local_position < max_shares and time_since_last_trade > 5:
            log_queue.put(f"📈 Price Up {price_change}% - BUY {quantity} Shares!")
            success, msg = place_bracket_order(symbol, quantity, latest_price, api_key, secret_key)
            log_queue.put(msg)
            if success:
                local_position += quantity
                last_trade_time = current_time
        
        elif price_change < -threshold and local_position >= quantity:
            if simulate:
                log_queue.put(f"📉 Simulated SELL {quantity} shares at {latest_price}")
                local_position -= quantity
            else:
                log_queue.put(f"📉 Price Down {price_change}% - SELL {quantity} Shares!")
                success, msg = place_order(symbol, quantity, "sell", api_key, secret_key)
                log_queue.put(msg)
                if success:
                    local_position -= quantity
                    last_trade_time = current_time
        else:
            log_queue.put(f"Debug: No trade - Change {price_change}% below threshold {threshold}% or position {local_position}/{max_shares}")

        prev_price = latest_price
        time.sleep(0.5)

# UI Controls
col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 Start Trading Bot", disabled=st.session_state.bot_active):
        if not alpaca_api_key or not alpaca_secret_key:
            st.session_state.logs.append("⚠ Please enter your Alpaca API credentials!")
        else:
            st.session_state.bot_active = True
            bot_running_event.set()
            thread = threading.Thread(
                target=trading_bot,
                args=(alpaca_api_key, alpaca_secret_key, symbol, price_threshold, trade_quantity, bot_running_event, simulate_prices, max_shares),
                daemon=True
            )
            thread.start()
with col2:
    if st.button("🛑 Stop Trading Bot", disabled=not st.session_state.bot_active):
        st.session_state.bot_active = False
        bot_running_event.clear()
        log_queue.put("🛑 Trading Bot Stopped.")

# Log display (updated on script rerun)
st.subheader("Activity Log")
log_placeholder = st.empty()

# Update logs in the main thread
while not log_queue.empty():
    log_entry = log_queue.get_nowait()
    st.session_state.logs.append(log_entry)

with log_placeholder.container():
    for entry in st.session_state.logs[-10:]:  # Show last 10 logs
        st.write(entry)