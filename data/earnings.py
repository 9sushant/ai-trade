"""Earnings calendar — avoids trades close to earnings announcements."""
from __future__ import annotations
from datetime import datetime, date, timedelta
from utils.logger import logger


class EarningsCalendar:
    _CACHE_TTL = 86400   # 24 hours

    def __init__(self):
        self._cache: dict[str, tuple[date | None, datetime]] = {}

    def get_earnings_date(self, symbol: str) -> date | None:
        cached = self._cache.get(symbol)
        if cached:
            earnings_date, ts = cached
            if (datetime.now() - ts).seconds < self._CACHE_TTL:
                return earnings_date
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            cal    = ticker.calendar
            if cal is not None and not cal.empty:
                for col in cal.columns:
                    if "earnings" in str(col).lower():
                        val = cal[col].iloc[0]
                        if hasattr(val, "date"):
                            earnings_date = val.date()
                            self._cache[symbol] = (earnings_date, datetime.now())
                            return earnings_date
        except Exception as exc:
            logger.debug(f"EarningsCalendar fetch {symbol}: {exc}")
        self._cache[symbol] = (None, datetime.now())
        return None

    def is_near_earnings(self, symbol: str, days_ahead: int = 5) -> bool:
        ed = self.get_earnings_date(symbol)
        if ed is None:
            return False
        today = date.today()
        return today <= ed <= today + timedelta(days=days_ahead)

    def should_avoid_trade(self, symbol: str, direction: str = "BUY") -> bool:
        ed = self.get_earnings_date(symbol)
        if ed is None:
            return False
        today = date.today()
        # Avoid 3 days before and 2 days after earnings
        return (today + timedelta(days=-2)) <= ed <= (today + timedelta(days=3))

    def get_upcoming_events(self, symbols: list[str]) -> dict[str, date | None]:
        return {s: self.get_earnings_date(s) for s in symbols}
