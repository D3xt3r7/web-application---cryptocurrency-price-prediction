from flask import Flask, render_template, request
from model import predict_price, create_chart

app = Flask(__name__)

coins = {
    'bitcoin': 'BTC/USDT',
    'ethereum': 'ETH/USDT',
    'binancecoin': 'BNB/USDT',
    'ripple': 'XRP/USDT',
    'cardano': 'ADA/USDT'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        coin_id = request.form['coin_id']
        symbol = coins.get(coin_id, 'BTC/USDT')
        
        market_data, pred_price, current_price, predictions, actuals, mae, rmse = predict_price(symbol=symbol, coin_id=coin_id)
        
        chart_config_price, chart_config_rsi, chart_config_atr, current_price, pred_price, rsi, atr = create_chart(
            market_data, pred_price, current_price, predictions, actuals
        )
        
        return render_template('results.html',
                             coin_id=coin_id,
                             symbol=symbol,
                             current_price=current_price,
                             pred_price=pred_price,
                             mae=mae,
                             rmse=rmse,
                             rsi=rsi,
                             atr=atr,
                             chart_config_price=chart_config_price,
                             chart_config_rsi=chart_config_rsi,
                             chart_config_atr=chart_config_atr)
    
    return render_template('index.html', coins=coins)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)