# config.py
USER_NAME = ""
PASSWORD = ""
API_KEY = ""
API_SECRET = ""
ACCESS_TOKEN = ""
REQUEST_TOKEN = ""
INSTRUMENT_TOKEN = 3486721 #256265  # Example: NIFTY50
INSTRUMENT_TOKENS = [10925826]
TELEGRAM_BOT_TOKEN = "" # Example: "1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAT_IDS = []
RISK = 0.001  # Example: Initial capital for backtesting
#Stocks and Tokens
INSTRUMENTS = {
    "NIFTY 50": 256265,  # NIFTY 50
    "BANKNIFTY": 260105,  # BANKNIFTY
    "SENSEX": 265,     # SENSEX
    "RELIANCE": 738561,   # RELIANCE
    "TCS": 5633,          # TCS
    "INFY": 408065,       # INFOSYS
    "HDFCBANK": 340481,   # HDFC Bank
    "SBIN": 779521,        # State Bank of India
    "Anantraj": 3486721,  # Anantraj Industries
}
# Option Chain URLs
option_chain_url_dict = {
    "nifty50": "https://kite.zerodha.com/markets/option-chain/INDICES/NIFTY%2050/256265",
    "niftybank": "https://kite.zerodha.com/markets/option-chain/INDICES/BANKNIFTY/260105",
    "sensex": "https://kite.zerodha.com/markets/option-chain/INDICES/SENSEX/265",
    "idea": "https://kite.zerodha.com/markets/option-chain/NSE/IDEA/3677697"
}
