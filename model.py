from twikit import Client
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ccxt
import requests
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from dotenv import load_dotenv
import os
from fredapi import Fred
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from telegram import Bot
import asyncio
from pycoingecko import CoinGeckoAPI

# Load environment variables from api.env
load_dotenv('/Users/piotr/Desktop/crypto_app/api.env')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
FRED_API_KEY = os.getenv('FRED_API_KEY')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
X_USERNAME = os.getenv('X_USERNAME')
X_EMAIL = os.getenv('X_EMAIL')
X_PASSWORD = os.getenv('X_PASSWORD')
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY')

def fetch_market_data(symbol='BTC/USDT', timeframe='1d', limit=200):
    exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_SECRET_KEY,
        'enableRateLimit': True
    })
    try:
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol  # Add symbol for chart labels

        # Calculate technical indicators
        df['rsi'] = RSIIndicator(df['close']).rsi()
        bb = BollingerBands(df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        atr = AverageTrueRange(df['high'], df['low'], df['close'])
        df['atr'] = atr.average_true_range()

        # Fetch ETH data for correlation
        eth_ohlcv = exchange.fetch_ohlcv('ETH/USDT', timeframe, limit)
        eth_df = pd.DataFrame(eth_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        eth_df['timestamp'] = pd.to_datetime(eth_df['timestamp'], unit='ms')
        df['eth_close'] = eth_df['close']

        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        print(f"fetch_market_data: Latest price for {symbol}: {df['close'].iloc[-1]}")
        print(f"fetch_market_data: Latest timestamp: {df['timestamp'].iloc[-1]}")
        print(f"fetch_market_data: RSI tail: {df['rsi'].tail(5)}")
        return df
    except Exception as e:
        print(f"Error fetching Binance data: {e}")
        # Return fallback DataFrame with realistic 2025 prices
        return pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(limit)],
            'open': [800.0 if 'BNB' in symbol else 60000.0] * limit,
            'high': [800.0 if 'BNB' in symbol else 60000.0] * limit,
            'low': [800.0 if 'BNB' in symbol else 60000.0] * limit,
            'close': [800.0 if 'BNB' in symbol else 60000.0] * limit,
            'volume': [0.0] * limit,
            'rsi': [50.0] * limit,
            'bb_high': [820.0 if 'BNB' in symbol else 62000.0] * limit,
            'bb_low': [780.0 if 'BNB' in symbol else 58000.0] * limit,
            'atr': [10.0 if 'BNB' in symbol else 1000.0] * limit,
            'eth_close': [2000.0] * limit,
            'symbol': [symbol] * limit
        })

def fetch_macro_data():
    try:
        fred = Fred(api_key=FRED_API_KEY)
        dxy = fred.get_series('DTWEXBGS').tail(1).iloc[0] if not fred.get_series('DTWEXBGS').tail(1).empty else 100.0
        cpi = fred.get_series('CPIAUCSL').tail(1).iloc[0] if not fred.get_series('CPIAUCSL').tail(1).empty else 3.0
        fed_rate = fred.get_series('FEDFUNDS').tail(1).iloc[0] if not fred.get_series('FEDFUNDS').tail(1).empty else 5.0
        print(f"fetch_macro_data: {dxy=}, {cpi=}, {fed_rate=}")
        return {'dxy': dxy, 'cpi': cpi, 'fed_rate': fed_rate}
    except Exception as e:
        print(f"Error fetching FRED data: {e}")
        return {'dxy': 100.0, 'cpi': 3.0, 'fed_rate': 5.0}

def create_chart(market_data, pred_price, current_price, predictions, actuals):
    print("create_chart: market_data['rsi'] tail:", market_data['rsi'].tail(5))
    print("create_chart: rsi single value:", market_data['rsi'].iloc[-1])
    dates = market_data['timestamp'].tail(30).astype(str).tolist()
    prices = market_data['close'].tail(30).tolist()
    rsi = market_data['rsi'].tail(30).tolist()
    atr = market_data['atr'].tail(30).tolist()
    pred_dates = [dates[-1], (pd.to_datetime(dates[-1]) + pd.Timedelta(days=7)).strftime('%Y-%m-%d')]
    pred_prices = [prices[-1], pred_price]
    
    chart_config_price = {
        "type": "line",
        "data": {
            "labels": dates + [pred_dates[1]],
            "datasets": [
                {
                    "label": f"Cena {market_data['symbol'].iloc[0]} (USD)",
                    "data": prices + [None],
                    "borderColor": "#1E90FF",
                    "fill": False,
                    "yAxisID": "y"
                },
                {
                    "label": "Predykcja",
                    "data": [None] * len(prices) + [pred_price],
                    "borderColor": "#FF4500",
                    "borderDash": [5, 5],
                    "fill": False,
                    "yAxisID": "y"
                }
            ]
        },
        "options": {
            "responsive": True,
            "scales": {
                "x": {"title": {"display": True, "text": "Data"}},
                "y": {"title": {"display": True, "text": "Cena (USD)"}, "position": "left"}
            },
            "plugins": {"title": {"display": True, "text": f"Predykcja ceny {market_data['symbol'].iloc[0]}"}}
        }
    }
    
    chart_config_rsi = {
        "type": "line",
        "data": {
            "labels": dates,
            "datasets": [
                {
                    "label": "RSI",
                    "data": rsi,
                    "borderColor": "#32CD32",
                    "fill": False
                }
            ]
        },
        "options": {
            "responsive": True,
            "scales": {
                "x": {"title": {"display": True, "text": "Data"}},
                "y": {"title": {"display": True, "text": "RSI"}, "min": 0, "max": 100}
            },
            "plugins": {"title": {"display": True, "text": "Wskaźnik RSI"}}
        }
    }
    
    chart_config_atr = {
        "type": "line",
        "data": {
            "labels": dates,
            "datasets": [
                {
                    "label": "ATR",
                    "data": atr,
                    "borderColor": "#FFD700",
                    "fill": False
                }
            ]
        },
        "options": {
            "responsive": True,
            "scales": {
                "x": {"title": {"display": True, "text": "Data"}},
                "y": {"title": {"display": True, "text": "ATR (USD)"}, "min": 0}
            },
            "plugins": {"title": {"display": True, "text": "Wskaźnik ATR"}}
        }
    }
    
    rsi_value = market_data['rsi'].iloc[-1] if not pd.isna(market_data['rsi'].iloc[-1]) else 50.0
    atr_value = market_data['atr'].iloc[-1] if not pd.isna(market_data['atr'].iloc[-1]) else 0.0
    print("create_chart: Returning rsi_value:", rsi_value)
    return chart_config_price, chart_config_rsi, chart_config_atr, current_price, pred_price, rsi_value, atr_value

def predict_price(symbol='BTC/USDT', coin_id='bitcoin'):
    try:
        # Fetch market data
        market_data = fetch_market_data(symbol=symbol)
        print(f"predict_price: market_data shape: {market_data.shape}")
        print(f"predict_price: market_data close tail: {market_data['close'].tail(5)}")

        # Fetch macro data
        macro_data = fetch_macro_data()
        print(f"predict_price: macro_data: {macro_data}")

        # Prepare data for LSTM
        features = ['close', 'rsi', 'bb_high', 'bb_low', 'atr', 'eth_close']
        data = market_data[features].copy()
        data['dxy'] = macro_data['dxy']
        data['cpi'] = macro_data['cpi']
        data['fed_rate'] = macro_data['fed_rate']
        data = data.fillna(method='ffill').fillna(method='bfill')
        print(f"predict_price: data shape after macro: {data.shape}")
        print(f"predict_price: data tail: {data.tail(5)}")

        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        if np.any(np.isnan(scaled_data)):
            print("predict_price: NaN detected in scaled_data")
            scaled_data = np.nan_to_num(scaled_data, nan=0.0)

        # Prepare sequences
        sequence_length = 60
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 0])  # Predict 'close'
        X, y = np.array(X), np.array(y)
        print(f"predict_price: X shape: {X.shape}, y shape: {y.shape}")

        # Check for empty or invalid data
        if X.shape[0] == 0 or y.shape[0] == 0:
            print("predict_price: No valid sequences for training")
            current_price = market_data['close'].iloc[-1]
            return market_data, current_price * 1.05, current_price, [], [], 0.0, 0.0

        # Split data
        train_size = int(len(X) * 0.8)
        if train_size == 0:
            print("predict_price: Not enough data for training")
            current_price = market_data['close'].iloc[-1]
            return market_data, current_price * 1.05, current_price, [], [], 0.0, 0.0
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build and train model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Predict
        last_sequence = scaled_data[-sequence_length:]
        last_sequence = last_sequence.reshape((1, sequence_length, len(features) + 3))
        pred_scaled = model.predict(last_sequence, verbose=0)
        pred_price_scaled = np.array([[pred_scaled[0][0]] + [0] * (len(features) + 2)])
        pred_price = scaler.inverse_transform(pred_price_scaled)[0][0]
        if pd.isna(pred_price):
            print("predict_price: NaN prediction detected, using fallback")
            pred_price = market_data['close'].iloc[-1] * 1.05
        print(f"predict_price: pred_price: {pred_price}")

        # Evaluate
        predictions = model.predict(X_test, verbose=0) if X_test.size > 0 else np.array([])
        actuals = y_test if y_test.size > 0 else np.array([])
        mae = np.mean(np.abs(predictions - actuals)) if predictions.size > 0 else 0.0
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2)) if predictions.size > 0 else 0.0
        print(f"predict_price: MAE: {mae}, RMSE: {rmse}")

        # Current price
        current_price = market_data['close'].iloc[-1]
        print(f"predict_price: current_price: {current_price}")

        # Send Telegram notification
        async def send_notification():
            bot = Bot(token=TELEGRAM_BOT_TOKEN)
            message = (f"KUPNO {symbol}: Cena {current_price:.2f}, "
                      f"Predykcja {pred_price:.2f}, "
                      f"Stop-loss {current_price * 0.95:.2f}, "
                      f"Take-profit {current_price * 1.1:.2f}")
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            print("Telegram notification sent")
        asyncio.run(send_notification())

        return market_data, pred_price, current_price, predictions, actuals, mae, rmse
    except Exception as e:
        print(f"Error in predict_price: {e}")
        current_price = market_data['close'].iloc[-1] if 'market_data' in locals() else 800.0
        return market_data, current_price * 1.05, current_price, [], [], 0.0, 0.0
def execute_trade(symbol, side, quantity):
    exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_SECRET_KEY,
        'enableRateLimit': True
    })
    try:
        order = exchange.create_market_order(symbol, side, quantity)
        print(f"Trade executed: {side} {quantity} {symbol}")
        return order
    except Exception as e:
        print(f"Error executing trade: {e}")
        return None