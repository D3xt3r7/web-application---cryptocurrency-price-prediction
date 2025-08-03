from telegram import Bot
import asyncio

async def test_telegram():
    bot = Bot(token='8222019926:AAFRMAPFFADKZhiQUD6751t-Cg_IWmleG7g')
    await bot.send_message(chat_id='8247636582', text='Testowa wiadomość')
    print("Wiadomość wysłana!")

asyncio.run(test_telegram())