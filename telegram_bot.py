from telegram.ext import Application, CommandHandler, MessageHandler, filters
from model import predict_price, create_chart

async def start(update, context):
    await update.message.reply_text('Witaj! Napisz "predict <coin>" (np. "predict bitcoin") lub użyj /predict.')

async def predict(update, context):
    coin = context.args[0].lower() if context.args else 'bitcoin'
    coins = {'bitcoin': 'BTC/USDT', 'ethereum': 'ETH/USDT', 'binancecoin': 'BNB/USDT', 'ripple': 'XRP/USDT', 'cardano': 'ADA/USDT'}
    symbol = coins.get(coin, 'BTC/USDT')
    
    try:
        market_data, pred_price, current_price, _, _, mae, rmse = predict_price(symbol=symbol, coin_id=coin)
        message = f"Predykcja dla {coin.capitalize()} ({symbol}):\nCena aktualna: {current_price:.2f}\nPredykcja: {pred_price:.2f}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}"
        await update.message.reply_text(message)
    except Exception as e:
        await update.message.reply_text(f"Błąd: {e}")

async def text_handler(update, context):
    text = update.message.text.lower()
    if text.startswith('predict '):
        coin = text.split(' ')[1]
        coins = {'bitcoin': 'BTC/USDT', 'ethereum': 'ETH/USDT', 'binancecoin': 'BNB/USDT', 'ripple': 'XRP/USDT', 'cardano': 'ADA/USDT'}
        symbol = coins.get(coin, 'BTC/USDT')
        
        try:
            market_data, pred_price, current_price, _, _, mae, rmse = predict_price(symbol=symbol, coin_id=coin)
            message = f"Predykcja dla {coin.capitalize()} ({symbol}):\nCena aktualna: {current_price:.2f}\nPredykcja: {pred_price:.2f}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}"
            await update.message.reply_text(message)
        except Exception as e:
            await update.message.reply_text(f"Błąd: {e}")

def main():
    app = Application.builder().token('8222019926:AAFRMAPFFADKZhiQUD6751t-Cg_IWmleG7g').build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('predict', predict))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.run_polling()

if __name__ == '__main__':
    main()