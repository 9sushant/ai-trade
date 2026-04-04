import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from utils.logger import logger


class DataFetcher:
    """Fetches historical and live stock data from Yahoo Finance (NSE)."""

    def __init__(self):
        self._cache: dict[str, pd.DataFrame] = {}

    def get_historical_data(
        self, symbol: str, period: str = "6mo", interval: str = "1d"
    ) -> pd.DataFrame | None:
        cache_key = f"{symbol}_{period}_{interval}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            df.index = pd.to_datetime(df.index)
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df.columns = ["open", "high", "low", "close", "volume"]
            self._cache[cache_key] = df
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def get_intraday_data(self, symbol: str, interval: str = "5m") -> pd.DataFrame | None:
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            df = ticker.history(period="5d", interval=interval)
            if df.empty:
                return None
            df.index = pd.to_datetime(df.index)
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df.columns = ["open", "high", "low", "close", "volume"]
            return df
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return None

    def get_live_price(self, symbol: str) -> dict | None:
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            info = ticker.fast_info
            return {
                "symbol": symbol,
                "price": round(info.last_price, 2),
                "prev_close": round(info.previous_close, 2),
                "change_pct": round(
                    ((info.last_price - info.previous_close) / info.previous_close) * 100, 2
                ),
                "day_high": round(info.day_high, 2) if hasattr(info, "day_high") else None,
                "day_low": round(info.day_low, 2) if hasattr(info, "day_low") else None,
            }
        except Exception as e:
            logger.error(f"Error fetching live price for {symbol}: {e}")
            return None

    def get_options_chain(self, symbol: str) -> dict | None:
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            expiry_dates = ticker.options
            if not expiry_dates:
                return None
            # Get nearest expiry
            nearest_expiry = expiry_dates[0]
            chain = ticker.option_chain(nearest_expiry)
            return {
                "symbol": symbol,
                "expiry": nearest_expiry,
                "calls": chain.calls,
                "puts": chain.puts,
            }
        except Exception as e:
            logger.error(f"Error fetching options chain for {symbol}: {e}")
            return None

    def get_bulk_data(self, symbols: list[str], period: str = "6mo") -> dict[str, pd.DataFrame]:
        results = {}
        for symbol in symbols:
            data = self.get_historical_data(symbol, period=period)
            if data is not None:
                results[symbol] = data
        return results

    def clear_cache(self):
        self._cache.clear()
