from fredapi import Fred
import os
from dotenv import load_dotenv

load_dotenv('/Users/piotr/Desktop/crypto_app/api.env')
FRED_API_KEY = os.getenv('FRED_API_KEY')
fred = Fred(api_key=FRED_API_KEY)
dxy = fred.get_series('DTWEXBGS').tail(1).iloc[0]
print(f"DXY: {dxy}")