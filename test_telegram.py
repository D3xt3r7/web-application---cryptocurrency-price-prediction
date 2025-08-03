from telegram import Bot
import asyncio

async def test_telegram():
    bot = Bot(token='xxxx')
    await bot.send_message(chat_id='xxx', text='Testowa wiadomość')
    print("Wiadomość wysłana!")

asyncio.run(test_telegram())
