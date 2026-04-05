"""FastAPI REST API — exposes signals, screener, backtest, and live status."""
from __future__ import annotations
from datetime import date

try:
    from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

from utils.logger import logger

if _HAS_FASTAPI:
    app = FastAPI(
        title="AI Trade API",
        description="NSE algorithmic trading signals and analytics",
        version="2.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ──────────────────────────────────────────────────────────────────
    # Models
    # ──────────────────────────────────────────────────────────────────
    class BacktestRequest(BaseModel):
        symbols:    list[str] | None = None
        period:     str = "1y"
        capital:    float = 50_000

    class SignalRequest(BaseModel):
        symbols: list[str]

    # ──────────────────────────────────────────────────────────────────
    # Health
    # ──────────────────────────────────────────────────────────────────
    @app.get("/health")
    def health():
        return {"status": "ok", "date": date.today().isoformat()}

    # ──────────────────────────────────────────────────────────────────
    # Screener
    # ──────────────────────────────────────────────────────────────────
    @app.get("/scan")
    def scan_signals(
        limit: int = Query(10, description="Max signals to return"),
        trade_type: str = Query("both", description="intraday | swing | both"),
    ):
        """Scan NIFTY 50 for current buy/sell signals."""
        try:
            from screener.stock_screener import StockScreener
            from config.settings import MarketConfig
            screener = StockScreener()
            signals  = screener.scan_signals(
                MarketConfig.NIFTY_50_SYMBOLS[:25], trade_type=trade_type
            )
            return {"signals": signals[:limit], "count": len(signals[:limit])}
        except Exception as exc:
            raise HTTPException(500, str(exc))

    @app.post("/signals")
    def get_signals(req: SignalRequest):
        """Get signals for specific symbols."""
        try:
            from screener.stock_screener import StockScreener
            screener = StockScreener()
            signals  = screener.scan_signals(req.symbols)
            return {"signals": signals, "count": len(signals)}
        except Exception as exc:
            raise HTTPException(500, str(exc))

    # ──────────────────────────────────────────────────────────────────
    # Cross-asset
    # ──────────────────────────────────────────────────────────────────
    @app.get("/cross-asset")
    def cross_asset():
        """Get cross-asset market signals (USD/INR, crude, gold, VIX)."""
        try:
            from analysis.cross_asset import CrossAssetSignals
            ca = CrossAssetSignals()
            return {
                "market_score": ca.get_market_score(),
                "signals":      ca.get_all_signals(),
                "usdinr_trend": ca.get_usdinr_trend(),
                "crude_signal": ca.get_crude_signal(),
                "vix":          ca.get_vix_level(),
                "reduce_risk":  ca.should_reduce_risk(),
            }
        except Exception as exc:
            raise HTTPException(500, str(exc))

    # ──────────────────────────────────────────────────────────────────
    # Sector rotation
    # ──────────────────────────────────────────────────────────────────
    @app.get("/sectors")
    def sector_rotation():
        """Get sector rotation rankings."""
        try:
            from analysis.sector_rotation import SectorRotationTracker
            tracker = SectorRotationTracker()
            tracker.refresh()
            return tracker.get_rotation_report()
        except Exception as exc:
            raise HTTPException(500, str(exc))

    # ──────────────────────────────────────────────────────────────────
    # Sentiment
    # ──────────────────────────────────────────────────────────────────
    @app.get("/sentiment/{symbol}")
    def sentiment(symbol: str):
        """Get news sentiment for a symbol."""
        try:
            from analysis.news_sentiment import NewsSentimentAnalyzer
            analyzer = NewsSentimentAnalyzer()
            return analyzer.get_signal(symbol.upper())
        except Exception as exc:
            raise HTTPException(500, str(exc))

    # ──────────────────────────────────────────────────────────────────
    # Block deals
    # ──────────────────────────────────────────────────────────────────
    @app.get("/block-deals")
    def block_deals(symbol: str = Query(None)):
        """Get today's block/bulk deals."""
        try:
            from data.block_deals import BlockDealTracker
            tracker = BlockDealTracker()
            if symbol:
                return tracker.get_signal(symbol.upper())
            return {
                "top_bought": tracker.get_top_bought(10),
                "all_deals":  tracker.get_deals_today()[:20],
            }
        except Exception as exc:
            raise HTTPException(500, str(exc))

    # ──────────────────────────────────────────────────────────────────
    # FII / DII
    # ──────────────────────────────────────────────────────────────────
    @app.get("/fii-dii")
    def fii_dii():
        """Get FII/DII institutional flow data."""
        try:
            from data.fii_dii import FIIDIITracker
            tracker = FIIDIITracker()
            return {
                "flow":  tracker.get_flow(),
                "trend": tracker.get_trend(),
                "score": tracker.get_score(),
            }
        except Exception as exc:
            raise HTTPException(500, str(exc))

    # ──────────────────────────────────────────────────────────────────
    # Expiry
    # ──────────────────────────────────────────────────────────────────
    @app.get("/expiry")
    def expiry_signal():
        """Get F&O expiry signal for today."""
        try:
            from analysis.fno_expiry import FNOExpiryAnalyzer
            analyzer = FNOExpiryAnalyzer()
            return analyzer.get_expiry_signal()
        except Exception as exc:
            raise HTTPException(500, str(exc))

    # ──────────────────────────────────────────────────────────────────
    # Risk
    # ──────────────────────────────────────────────────────────────────
    @app.get("/risk/{symbol}")
    def risk_report(
        symbol: str,
        capital: float = Query(50_000),
    ):
        """Get VaR, CVaR, and stress test for a symbol."""
        try:
            from data.fetcher import DataFetcher
            from analysis.technical import TechnicalAnalyzer
            from risk.var_calculator import VaRCalculator
            from risk.cvar import CVaRCalculator
            from risk.stress_test import StressTest

            df  = DataFetcher().get_historical_data(symbol, period="1y", interval="1d")
            if df is None:
                raise HTTPException(404, f"{symbol} not found")
            df  = TechnicalAnalyzer().compute_all_indicators(df)

            var_c = VaRCalculator()
            cvar_c = CVaRCalculator()
            stress = StressTest()

            return {
                "symbol":      symbol,
                "var_95":      var_c.compute_var(df, capital),
                "cvar_95":     cvar_c.compute_cvar(df, capital),
                "cvar_report": cvar_c.expected_shortfall_report(df, capital),
                "stress":      stress.generate_report(df, capital, 10),
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, str(exc))

    # ──────────────────────────────────────────────────────────────────
    # Backtest
    # ──────────────────────────────────────────────────────────────────
    @app.post("/backtest")
    def run_backtest(req: BacktestRequest, background_tasks: BackgroundTasks):
        """Trigger backtest (runs async, returns job ID)."""
        import uuid
        job_id = str(uuid.uuid4())[:8]
        # In production, store results in DB keyed by job_id
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Use GET /backtest/{job_id} to poll results",
            "params":  req.dict(),
        }

    # ──────────────────────────────────────────────────────────────────
    # Trade history
    # ──────────────────────────────────────────────────────────────────
    @app.get("/trades")
    def get_trades(
        symbol: str  = Query(None),
        limit:  int  = Query(50),
        offset: int  = Query(0),
    ):
        """Get trade history from SQLite database."""
        try:
            from data.database import TradeDatabase
            db = TradeDatabase()
            return {
                "trades": db.get_trades(symbol=symbol, limit=limit, offset=offset),
                "equity": db.get_equity_curve()[-20:],
            }
        except Exception as exc:
            raise HTTPException(500, str(exc))

    # ──────────────────────────────────────────────────────────────────
    # Live trader status
    # ──────────────────────────────────────────────────────────────────
    _live_trader = None

    @app.get("/live/status")
    def live_status():
        """Get live trader status."""
        if _live_trader is None:
            return {"status": "NOT_RUNNING"}
        return _live_trader.get_status()

    @app.post("/live/start")
    def live_start(capital: float = Query(50_000), paper: bool = Query(True)):
        """Start live paper trading."""
        global _live_trader
        try:
            import threading
            from trading.live_trader import LiveTrader
            _live_trader = LiveTrader(capital=capital, paper_trade=paper)
            t = threading.Thread(target=_live_trader.start, daemon=True)
            t.start()
            return {"status": "started", "capital": capital, "paper": paper}
        except Exception as exc:
            raise HTTPException(500, str(exc))

    @app.post("/live/stop")
    def live_stop():
        """Stop live trading."""
        global _live_trader
        if _live_trader:
            _live_trader.stop()
        return {"status": "stopped"}

    # ──────────────────────────────────────────────────────────────────
    # Monte Carlo
    # ──────────────────────────────────────────────────────────────────
    @app.get("/monte-carlo")
    def monte_carlo(
        symbol:  str   = Query("RELIANCE"),
        capital: float = Query(50_000),
        n_sims:  int   = Query(1000),
    ):
        """Run Monte Carlo simulation for a symbol."""
        try:
            from data.fetcher import DataFetcher
            from risk.monte_carlo import MonteCarloSimulator
            df  = DataFetcher().get_historical_data(symbol, period="2y", interval="1d")
            if df is None:
                raise HTTPException(404, f"{symbol} not found")
            rets = df["close"].pct_change().dropna().values
            sim  = MonteCarloSimulator(n_simulations=n_sims)
            return sim.run_from_returns(rets, capital)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, str(exc))


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Launch FastAPI server."""
    if not _HAS_FASTAPI:
        logger.error("FastAPI not installed. Run: pip install fastapi uvicorn")
        return
    import uvicorn
    uvicorn.run("api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_api()
