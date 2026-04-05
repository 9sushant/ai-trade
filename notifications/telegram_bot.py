"""
Telegram bot for real-time trade alerts and daily summaries.
Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable.
"""
from __future__ import annotations
import os
import time
import requests
from datetime import datetime
from utils.logger import logger


class TelegramNotifier:
    _API = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, token: str = None, chat_id: str = None):
        self.token   = token   or os.getenv("TELEGRAM_BOT_TOKEN",  "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID",    "")
        self._last_send = 0.0

    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def send_message(self, text: str) -> bool:
        if not self.is_configured():
            logger.debug("Telegram not configured — skipping notification")
            return False
        # Rate limit: 1 msg/sec
        elapsed = time.time() - self._last_send
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        try:
            resp = requests.post(
                self._API.format(token=self.token),
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            self._last_send = time.time()
            return resp.ok
        except Exception as exc:
            logger.debug(f"Telegram send failed: {exc}")
            return False

    def send_signal(self, signal: dict) -> bool:
        sym  = signal.get("symbol", "?")
        rec  = signal.get("recommendation", signal.get("direction", "?"))
        px   = signal.get("close", signal.get("entry", 0))
        sl   = signal.get("stop_loss", 0)
        tgt  = signal.get("target", signal.get("target_1", 0))
        score= signal.get("composite_score", signal.get("score", 0))
        ml   = signal.get("ml_prob", "")
        hold = signal.get("hold_days", signal.get("hold", ""))
        emoji = "🟢" if "BUY" in str(rec).upper() else "🔴"
        ml_str = f" | ML: {ml:.0%}" if isinstance(ml, float) else ""
        hold_str = f"\n⏰ Hold: {hold}" if hold else ""
        msg = (f"{emoji} <b>SIGNAL: {rec} {sym}</b>\n"
               f"💰 Entry: ₹{px:,.2f}\n"
               f"🛑 SL: ₹{sl:,.2f}\n"
               f"🎯 Target: ₹{tgt:,.2f}\n"
               f"⚡ Score: {score}{ml_str}{hold_str}")
        return self.send_message(msg)

    def send_trade_opened(self, trade: dict) -> bool:
        sym = trade.get("symbol", "?")
        dir = trade.get("direction", "?")
        qty = trade.get("quantity", 0)
        px  = trade.get("entry_price", 0)
        sl  = trade.get("stop_loss", px)
        tgt = trade.get("target", px)
        emoji = "✅🟢" if dir == "BUY" else "✅🔴"
        return self.send_message(
            f"{emoji} <b>TRADE OPENED: {dir} {qty}×{sym}</b>\n"
            f"📈 Entry: ₹{px:,.2f}\n"
            f"🛑 SL: ₹{sl:,.2f} | 🎯 TGT: ₹{tgt:,.2f}"
        )

    def send_trade_closed(self, trade: dict, running_pnl: float = 0) -> bool:
        sym    = trade.get("symbol", "?")
        dir    = trade.get("direction", "?")
        pnl    = trade.get("pnl", 0)
        pnl_pct= trade.get("pnl_pct", 0)
        reason = trade.get("exit_reason", "?")
        emoji  = "💰" if pnl > 0 else "💸"
        sign   = "+" if pnl >= 0 else ""
        return self.send_message(
            f"{emoji} <b>TRADE CLOSED: {sign}₹{pnl:,.0f} ({sign}{pnl_pct:.1f}%)</b>\n"
            f"📊 {sym} {dir} | Exit: {reason}\n"
            f"📈 Running P&L: ₹{running_pnl:,.0f}"
        )

    def send_daily_summary(self, stats: dict) -> bool:
        date    = datetime.now().strftime("%d %b %Y")
        pnl     = stats.get("daily_pnl", 0)
        trades  = stats.get("trades_today", 0)
        wr      = stats.get("win_rate", 0)
        mdd     = stats.get("max_drawdown_pct", 0)
        regime  = stats.get("regime", "?")
        emoji   = "📈" if pnl >= 0 else "📉"
        sign    = "+" if pnl >= 0 else ""
        reg_icon= {"BULL":"🐂","BEAR":"🐻","SIDEWAYS":"➡️"}.get(regime, "⚪")
        return self.send_message(
            f"📊 <b>DAILY SUMMARY — {date}</b>\n"
            f"{emoji} Trades: {trades} | P&L: {sign}₹{pnl:,.0f}\n"
            f"🏆 Win Rate: {wr:.1f}% | 📉 MDD: {mdd:.1f}%\n"
            f"{reg_icon} Regime: {regime}"
        )

    def send_market_regime(self, regime: str) -> bool:
        icons = {"BULL": "🐂 BULL", "BEAR": "🐻 BEAR", "SIDEWAYS": "➡️ SIDEWAYS"}
        return self.send_message(f"🌍 <b>Market Regime: {icons.get(regime, regime)}</b>")

    def send_error(self, message: str) -> bool:
        return self.send_message(f"⚠️ <b>ERROR</b>\n{message}")

    def send_retrain_complete(self, stats: dict) -> bool:
        n       = stats.get("n_samples", 0)
        pos_r   = stats.get("pos_rate", 0)
        models  = stats.get("models", [])
        return self.send_message(
            f"🤖 <b>ML Model Retrained</b>\n"
            f"📊 Samples: {n} | Win rate: {pos_r:.1%}\n"
            f"🧠 Models: {', '.join(models)}"
        )
