import os
from dotenv import find_dotenv, load_dotenv
from alpaca_trade_api.rest import REST

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

endpoint = os.getenv("alpaca_endpoint")
API_KEY = os.getenv("alpaca_key")
API_SECRET_KEY = os.getenv("alpaca_secret")

api = REST(
    key_id = API_KEY,
    secret_key=API_SECRET_KEY,
    base_url=endpoint
)

account = api.get_account()
print("Status:", account.status)
print("Cash:", account.cash)