Crypto Price Prediction and Trading Bot

Welcome to the Crypto Price Prediction and Trading Bot repository! This project is a comprehensive solution for cryptocurrency price prediction, market analysis, and automated trading, built with Python. It integrates real-time data from Binance, macroeconomic indicators from the Federal Reserve Economic Data (FRED), and a Telegram bot for notifications, all wrapped in a Flask web application.

Overview

This application leverages machine learning (LSTM models) to predict cryptocurrency prices, calculates technical indicators (RSI, Bollinger Bands, ATR), and executes trades on Binance. It also fetches macroeconomic data (DXY, CPI, Fed Rate) to enhance prediction accuracy. The Telegram bot provides real-time updates, while the Flask app offers a user-friendly interface to visualize predictions and market trends.

Features





Price Prediction: Uses LSTM neural networks to forecast cryptocurrency prices (e.g., BTC/USDT, ETH/USDT).



Technical Analysis: Calculates RSI, Bollinger Bands, and ATR for market insights.



Macroeconomic Integration: Incorporates DXY, CPI, and Fed Rate from FRED for contextual analysis.



Automated Trading: Executes market orders on Binance based on predictions.



Telegram Notifications: Sends price predictions and trade signals via a Telegram bot.



Web Interface: Flask-based app with charts for price, RSI, and ATR visualization.

Installation





Clone the Repository:

git clone https://github.com/D3xt3r7/crypto-prediction-bot.git
cd crypto-prediction-bot



Set Up Environment:





Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



Install Dependencies:





Install required packages from requirements.txt:

pip install -r requirements.txt



Configure API Keys:





Create a file named api.env in the project root (e.g., /Users/piotr/Desktop/crypto_app/api.env).



Add your API keys and credentials:

BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
FRED_API_KEY=your_fred_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
X_USERNAME=your_x_username
X_EMAIL=your_x_email
X_PASSWORD=your_x_password
COINGECKO_API_KEY=your_coingecko_api_key



Ensure the .env file is added to .gitignore to keep credentials secure.



Run the Application:



Start the Flask web app:

python app.py



Access it at http://0.0.0.0:5001 (default port).



Test the Telegram bot:

python test_telegram.py

Project Structure



app.py: Flask application entry point with routes for the web interface.



model.py: Core logic for data fetching, price prediction, and chart generation.



telegram_bot.py: Telegram bot implementation with prediction commands.



test_fred.py: Script to test FRED API integration.



test_telegram.py: Script to test Telegram bot functionality.



api.env: Environment file for API keys (not included in repo).



requirements.txt: List of Python dependencies.

Usage



Web Interface: Visit the app, select a coin (Bitcoin, Ethereum, etc.), and view predictions with charts.



Telegram Bot: Use commands like /start or type predict <coin> (e.g., predict bitcoin) to get price forecasts.



Trading: The execute_trade function in model.py can be triggered manually or integrated into the bot for automation.

Technical Details



Libraries:



ccxt for Binance API interactions.



fredapi for FRED macroeconomic data.



tensorflow and sklearn for LSTM-based predictions.



ta for technical indicators.



flask for the web app.



python-telegram-bot for bot functionality.



twikit for Twitter/X integration (not fully implemented in provided code).



Data Sources:


Binance for OHLCV data.



FRED for DXY, CPI, and Fed Rate.



CoinGecko API (via pycoingecko) as a fallback.



Prediction Model:


LSTM network with 50 units, Dropout (0.2), and 10 epochs.



Features include close price, RSI, Bollinger Bands, ATR, ETH correlation, and macro data.



Charts: Generated using Chart.js configuration objects for price, RSI, and ATR visualization.

Contributing

Feel free to fork this repository, submit pull requests, or report issues. Suggestions for improving the model, adding new coins, or enhancing the UI are welcome!

License

This project is for educational purposes only. Use the code responsibly and respect the terms of service of all APIs used.

