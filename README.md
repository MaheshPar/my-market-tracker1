# LOFO Terminal (Zerodha Trade Tracker)

A high-performance algorithmic trading and portfolio tracking dashboard for Indian Equity markets (NSE/BSE), featuring **LOFO (Lowest In, First Out)** lot matching, tranche intelligence, technical indicator radar, swing pivot signals, and state export.

## Key Features

- **LOFO Engine**: Matches exits against lowest buy price lots first, calculating true operational holding cost, cashflows, and date-weighted XIRR.
- **Capital Block Tranching**: Automatically splits open positions into capital allocation tranches (configurable, default ₹8,000) with peak close drawdown tracking.
- **High-Performance Caching Layer**: Uses `@st.cache_data` with granular TTLs:
  - Watchlists: 10m TTL with automatic local file fallback.
  - Historical Daily Candles & Technical Indicators: 12h TTL.
  - Real-Time Live LTP: 60s TTL with non-destructive cache invalidation.
- **Refactored Sidebar Navigation**:
  - Live market status indicator with IST clock (trading hours 09:15 - 15:30 IST).
  - Quick portfolio summary glance (Capital Deployed, Current Valuation, Unrealized P&L).
  - Clean view switching between Portfolio, Watchlists, Technical Charts, Closed Trades, Swing Radar, Trade Uploads, and Data Export.
  - Terminal Settings: Theme switcher (Dark/Light), auto-refresh controls, and block size configuration.
- **Zero-Crash Resilience & Fallbacks**:
  - Pure Python bisection XIRR fallback when `scipy` is not installed.
  - Native line chart fallback if `plotly` is not installed.
  - Graceful fallback to local watchlists and tradebooks when GitHub secrets are not configured.
- **Direct Excel Export**: Export full terminal state (Holdings, Blocks, Closed Trades, Indicators, Swing Radar) directly to browser or push to GitHub.

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Application
```bash
streamlit run app.py
```

### 3. Run Unit Tests
```bash
# Using standard Python unittest (no extra packages needed):
python -m unittest discover -s tests -p "test_*.py" -v

# Or using pytest:
pytest tests/ -v
```

---

## Directory Structure

```
my-market-tracker/
├── app.py                      # Main Streamlit dashboard with sidebar navigation
├── utils/
│   ├── data_processing.py      # Core quantitative algorithms (LOFO, XIRR, RSI, scoring, pivots)
│   └── kiteapp.py              # Zerodha Kite Connect utilities
├── tests/
│   └── test_data_processing.py # 27 comprehensive unit tests
├── requirements.txt            # Project dependencies
├── tradebook.csv               # Master Zerodha tradebook
├── manual_trades.csv           # Pending manual emergency trades
├── stock_watchlist.txt         # Stock tickers
├── etf_watchlist.txt           # ETF tickers
├── nifty50_watchlist.txt       # Nifty 50 tickers
└── niftynext50_watchlist.txt   # Nifty Next 50 tickers
```

---

## Secrets Configuration (Optional)

In `.streamlit/secrets.toml` or environment variables:
```toml
GITHUB_TOKEN = "your_github_personal_access_token"
GITHUB_REPO  = "your_username/your_repo"
```
*Note: If secrets are omitted, the app automatically runs in **Local Files Mode**.*
