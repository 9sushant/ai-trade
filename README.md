# AI Trade - Smart Stock Trading System

An automated stock trading system for Indian markets (NSE/BSE) with:
- **Technical Analysis**: RSI, MACD, EMA crossovers, Bollinger Bands, SuperTrend, VWAP, ADX, Volume analysis
- **Top 10 Daily Picks**: Intraday + swing trade recommendations every morning
- **Options Analysis**: Call/Put recommendations based on directional signals
- **Angel One Integration**: Auto-place orders via SmartAPI
- **Risk Management**: Position sizing, stop-loss, ₹500/day profit target enforcement

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Credentials

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Angel One SmartAPI
ANGEL_API_KEY=your_api_key_here
ANGEL_CLIENT_ID=your_client_id_here
ANGEL_PASSWORD=your_mpin_here
ANGEL_TOTP_SECRET=your_totp_base32_secret_here

# Trading limits (adjust to your comfort level)
MAX_CAPITAL=50000
MAX_RISK_PER_TRADE=500
DAILY_PROFIT_TARGET=500
MAX_DAILY_LOSS=100
MAX_POSITIONS=5
```

**How to get Angel One SmartAPI credentials:**
1. Log in to [Angel One](https://www.angelone.in)
2. Go to **SmartAPI** → Create a new app
3. Copy the **API Key**
4. Your **Client ID** = your Angel One login ID
5. **Password** = your 4-digit MPIN
6. **TOTP Secret** = the base32 key shown when setting up TOTP in Angel One settings

### 3. Run the App

```bash
# Interactive mode (recommended for first run)
python main.py

# Just show Top 10 picks (no broker needed)
python main.py --top10

# Full scan with options analysis (no broker needed)
python main.py --scan

# Auto trading mode (places real orders)
python main.py --auto
```

---

## Modes of Operation

### `--scan` (No broker required)
Scans the market, runs full technical analysis, and prints:
- Top 10 intraday picks
- Top 10 swing trade picks
- Options recommendations

No Angel One credentials needed. Uses free data from Yahoo Finance (NSE).

### Interactive Mode (default)
Full interactive menu:
```
COMMANDS:
  1 - Show Top 10 Picks
  2 - Intraday Opportunities
  3 - Swing Trade Picks
  4 - Options Analysis
  5 - Risk Dashboard
  6 - Execute Recommendations (Manual confirmation)
  7 - Start Auto Trading Loop
  8 - Refresh Scan
  9 - Portfolio & Positions
  0 - Exit
```

### `--auto` (Auto Trading)
Fully automated trading loop:
- Runs morning scan at 9:15 AM
- Places bracket orders (entry + SL + target) automatically
- Monitors positions every 5 minutes
- Auto-exits at target/stop-loss
- Stops when ₹500 profit target OR ₹100 loss limit is hit
- Square-off all intraday positions by 3:15 PM

---

## How It Works

### Technical Analysis Engine

Every stock is analyzed using 7 indicator categories:

| Indicator | Signal |
|-----------|--------|
| EMA Crossover (9/21) | Trend direction change |
| MACD | Momentum confirmation |
| RSI (14) | Overbought/oversold (30/70) |
| Bollinger Bands | Volatility breakout |
| SuperTrend (10,3) | Trend filter |
| Volume Ratio | Trade confirmation |
| ADX | Trend strength (>25 = strong) |

**Composite Score**: -100 to +100
- Score ≥ 40: STRONG BUY
- Score ≥ 20: BUY
- Score ≤ -40: STRONG SELL
- Score ≤ -20: SELL

### Screener Logic

**Intraday picks** (Nifty 50 stocks only - high liquidity):
- ADX > 25 (strong trend)
- Volume ratio > 1.5x (unusual activity)
- Clear directional signal (score ≥ 30)

**Swing picks** (Nifty 50 + Nifty 200):
- RSI pullback in uptrend (35-50 in uptrend)
- Near support level
- SuperTrend bullish
- Hold: 3-7 days

### Risk Management

Per-trade rules:
- **Position size** = min(MAX_RISK / (entry - stop_loss), 30% of capital)
- **Stop loss**: 1.5x ATR below entry
- **Target**: 2x ATR above entry (minimum 1.5:1 R:R)
- **Max loss per trade**: ₹500 (configurable)

Daily rules:
- Stop trading when daily profit ≥ ₹500
- Stop trading when daily loss ≥ ₹100
- Max 5 open positions at once

### Options Analysis

For stocks with strong composite scores:
- **Bullish signal** → Buy ATM Call OR Sell OTM Put
- **Bearish signal** → Buy ATM Put OR Sell OTM Call
- **Neutral** → Straddle/Strangle evaluation

---

## Project Structure

```
ai-trade/
├── main.py                      # Entry point + CLI
├── config/
│   └── settings.py              # All configuration (from .env)
├── data/
│   └── fetcher.py               # NSE data via yfinance
├── analysis/
│   ├── technical.py             # RSI, MACD, EMA, BB, SuperTrend, VWAP
│   └── patterns.py              # Candlestick + chart pattern detection
├── screener/
│   └── stock_screener.py        # Top 10 intraday + swing picks
├── options/
│   └── analyzer.py              # Options chain + trade suggestions
├── broker/
│   └── angel_one.py             # Angel One SmartAPI wrapper
├── agents/
│   ├── scanner_agent.py         # Morning + live market scanner
│   ├── analysis_agent.py        # Deep technical analysis
│   ├── risk_agent.py            # Risk checks + position sizing
│   ├── executor_agent.py        # Order execution
│   └── trading_orchestrator.py  # Coordinates all agents
├── dashboard/
│   └── terminal_ui.py           # Rich terminal dashboard
└── utils/
    ├── logger.py                # Structured logging
    └── helpers.py               # Utility functions
```

---

## ⚠️ Important Disclaimers

1. **No profit guarantee**: While the system is designed to find high-probability setups, stock trading always involves risk. Past performance does not guarantee future results.

2. **Test in paper mode first**: By default the app runs in analysis-only mode. Before using `--auto`, paper trade manually for at least 2 weeks.

3. **Start small**: Use ₹10,000-20,000 capital initially. Increase only after consistent results.

4. **Market conditions matter**: This system works best in trending markets (ADX > 25). In sideways/choppy markets, reduce position sizes or stay out.

5. **You are responsible**: Always review recommendations before executing. Never let any automated system trade more than you can afford to lose.

---

## Logs

All activity is logged to `logs/` directory:
- `logs/trading_YYYY-MM-DD.log` - Full debug log
- `logs/trades_YYYY-MM-DD.log` - Trade-only log (entries, exits, P&L)

---

## Requirements

- Python 3.10+
- Angel One account with SmartAPI enabled
- Internet connection during market hours (9:15 AM - 3:30 PM IST)

## Support

For Angel One SmartAPI documentation: https://smartapi.angelone.in/
