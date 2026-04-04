import os
from dotenv import load_dotenv

load_dotenv()


class AngelOneConfig:
    API_KEY = os.getenv("ANGEL_API_KEY", "")
    CLIENT_ID = os.getenv("ANGEL_CLIENT_ID", "")
    PASSWORD = os.getenv("ANGEL_PASSWORD", "")
    TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET", "")


class TradingConfig:
    MAX_CAPITAL = float(os.getenv("MAX_CAPITAL", 50000))
    MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", 500))
    DAILY_PROFIT_TARGET = float(os.getenv("DAILY_PROFIT_TARGET", 500))
    MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", 1000))
    MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", 5))
    STOP_LOSS_PCT = 1.5  # 1.5% stop loss
    TARGET_PCT = 2.5  # 2.5% target
    TRAILING_SL_PCT = 0.5  # 0.5% trailing stop loss


class MarketConfig:
    MARKET_OPEN = "09:15"
    MARKET_CLOSE = "15:30"
    PRE_MARKET_SCAN = "09:00"
    EXCHANGE = "NSE"
    # Top liquid stocks for scanning
    NIFTY_50_SYMBOLS = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
        "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "BAJFINANCE",
        "TITAN", "SUNPHARMA", "ULTRACEMCO", "NESTLEIND", "WIPRO",
        "HCLTECH", "TATAMOTORS", "POWERGRID", "NTPC", "TATASTEEL",
        "JSWSTEEL", "TECHM", "ONGC", "INDUSINDBK", "HINDALCO",
        "ADANIENT", "ADANIPORTS", "DRREDDY", "DIVISLAB", "CIPLA",
        "BAJAJFINSV", "GRASIM", "BRITANNIA", "EICHERMOT", "HEROMOTOCO",
        "APOLLOHOSP", "COALINDIA", "BPCL", "TATACONSUM", "M&M",
        "HDFCLIFE", "SBILIFE", "UPL", "LTIM", "BAJAJ-AUTO",
    ]
    # Extended list for broader scanning
    NIFTY_200_EXTRA = [
        "ZOMATO", "DMART", "PIDILITIND", "HAVELLS", "GODREJCP",
        "VOLTAS", "TRENT", "PEL", "PIIND", "ASTRAL",
        "PERSISTENT", "COFORGE", "MPHASIS", "LTTS", "HAPPSTMNDS",
        "Dixon", "POLYCAB", "KAYNES", "BEL", "HAL",
        "IRFC", "PNB", "BANKBARODA", "CANBK", "IOB",
        "TATAPOWER", "NHPC", "SJVN", "RECLTD", "PFC",
    ]
    ALL_SCAN_SYMBOLS = NIFTY_50_SYMBOLS + NIFTY_200_EXTRA
