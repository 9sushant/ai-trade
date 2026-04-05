"""Real-time streaming data handler — WebSocket via Kite or polling via yfinance."""
from __future__ import annotations
import time
import threading
from datetime import datetime
from utils.logger import logger


class StreamingDataHandler:
    def __init__(self, symbols: list[str], on_tick_callback=None):
        self.symbols           = symbols
        self.on_tick_callback  = on_tick_callback
        self._latest:  dict[str, dict] = {}
        self._lock     = threading.Lock()
        self._running  = False
        self._thread:  threading.Thread | None = None

    # ------------------------------------------------------------------
    def start_polling_stream(self, interval_seconds: int = 5):
        """Poll yfinance every N seconds in a background thread."""
        self._running = True
        self._thread  = threading.Thread(
            target=self._poll_loop, args=(interval_seconds,), daemon=True
        )
        self._thread.start()
        logger.info(f"Polling stream started for {len(self.symbols)} symbols "
                    f"(interval={interval_seconds}s)")

    def start_kite_stream(self, kite_trader):
        """Use Kite WebSocket if kiteconnect available, else fall back to polling."""
        try:
            from kiteconnect import KiteTicker
            if not kite_trader._is_connected or kite_trader._kite is None:
                logger.warning("Kite not connected — falling back to polling")
                self.start_polling_stream()
                return
            tokens = self._get_instrument_tokens(kite_trader)
            ticker = KiteTicker(kite_trader.api_key,
                                kite_trader._kite.access_token)
            ticker.on_ticks   = self._on_kite_ticks
            ticker.on_connect = lambda ws, r: ws.subscribe(tokens)
            ticker.connect(threaded=True)
            self._running = True
            logger.info("Kite WebSocket stream started")
        except Exception as exc:
            logger.warning(f"Kite stream error ({exc}) — falling back to polling")
            self.start_polling_stream()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Streaming stopped")

    def get_latest(self, symbol: str) -> dict | None:
        with self._lock:
            return self._latest.get(symbol)

    def get_all_latest(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._latest)

    # ------------------------------------------------------------------
    def _poll_loop(self, interval: int):
        import yfinance as yf
        while self._running:
            for symbol in self.symbols:
                try:
                    t     = yf.Ticker(f"{symbol}.NS")
                    info  = t.fast_info
                    tick  = {
                        "symbol":      symbol,
                        "last_price":  getattr(info, "last_price",   0),
                        "open":        getattr(info, "open",          0),
                        "high":        getattr(info, "day_high",      0),
                        "low":         getattr(info, "day_low",       0),
                        "volume":      getattr(info, "three_month_average_volume", 0),
                        "timestamp":   datetime.now().isoformat(),
                    }
                    with self._lock:
                        self._latest[symbol] = tick
                    if self.on_tick_callback:
                        self.on_tick_callback(symbol, tick)
                except Exception:
                    pass
            time.sleep(interval)

    def _on_kite_ticks(self, ws, ticks):
        for tick in ticks:
            symbol = str(tick.get("tradingsymbol", ""))
            data   = {
                "symbol":     symbol,
                "last_price": tick.get("last_price", 0),
                "open":       tick.get("ohlc", {}).get("open", 0),
                "high":       tick.get("ohlc", {}).get("high", 0),
                "low":        tick.get("ohlc", {}).get("low",  0),
                "volume":     tick.get("volume", 0),
                "timestamp":  datetime.now().isoformat(),
            }
            with self._lock:
                self._latest[symbol] = data
            if self.on_tick_callback:
                self.on_tick_callback(symbol, data)

    def _get_instrument_tokens(self, kite_trader) -> list[int]:
        try:
            instruments = kite_trader._kite.instruments("NSE")
            tokens      = []
            for inst in instruments:
                if inst["tradingsymbol"] in self.symbols:
                    tokens.append(inst["instrument_token"])
            return tokens
        except Exception:
            return []
