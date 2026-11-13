import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import requests
import base64
import time
import io
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_processing import (
    calculate_xirr,
    get_xirr_weighted_buy_date,
    process_lofo,
    split_lots_into_blocks,
    calculate_wilder_rsi,
    parse_tradebook,
    compute_score,
    score_label,
    detect_swing_pivots,
    get_ist_now,
    is_market_open,
    apply_live_prices
)

try:
    from zoneinfo import ZoneInfo
except ImportError:
    try:
        from backports.zoneinfo import ZoneInfo
    except ImportError:
        ZoneInfo = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    make_subplots = None
    PLOTLY_AVAILABLE = False

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

warnings.filterwarnings('ignore')

# ─── CONFIG & SECRETS ─────────────────────────────────────────────────────────
DEFAULT_BLOCK_SIZE  = 8000
CSV_FILENAME        = "tradebook.csv"
MANUAL_CSV_FILENAME = "manual_trades.csv"
STOCK_FILE_PATH       = "stock_watchlist.txt"
ETF_FILE_PATH         = "etf_watchlist.txt"
NIFTY50_FILE_PATH     = "nifty50_watchlist.txt"
NIFTYNEXT50_FILE_PATH = "niftynext50_watchlist.txt"

# Safe secret retrieval with sandbox/file permission resilience
try:
    GITHUB_TOKEN = str(st.secrets.get("GITHUB_TOKEN", os.environ.get("GITHUB_TOKEN", "")) or "").strip()
    GITHUB_REPO  = str(st.secrets.get("GITHUB_REPO", os.environ.get("GITHUB_REPO", "")) or "").strip()
except Exception:
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
    GITHUB_REPO  = os.environ.get("GITHUB_REPO", "").strip()

DEFAULT_ETF_LIST = [
    "HNGSNGBEES","PVTBANIETF","AUTOBEES","JUNIORBEES","PHARMABEES",
    "MODEFENCE","MID150BEES","HDFCSML250","CPSEETF","MONQ50","MOCAPITAL"
]

st.set_page_config(
    page_title="LOFO Terminal",
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme default
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "refresh_interval" not in st.session_state:
    st.session_state.refresh_interval = 180

if "auto_refresh_enabled" not in st.session_state:
    st.session_state.auto_refresh_enabled = True

if "block_size" not in st.session_state:
    st.session_state.block_size = DEFAULT_BLOCK_SIZE

BLOCK_SIZE = st.session_state.block_size

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — THEME AWARE
# ══════════════════════════════════════════════════════════════════════════════
THEMES = {
    "dark": {
        "bg":       "#060608",
        "s1":       "#0e0f14",
        "s2":       "#14161e",
        "s3":       "#1c1f2b",
        "b":        "rgba(255,255,255,0.07)",
        "bh":       "rgba(255,255,255,0.14)",
        "bs":       "rgba(255,255,255,0.05)",
        "acc":      "#00e5ff",
        "acc_d":    "rgba(0,229,255,0.10)",
        "acc_g":    "rgba(0,229,255,0.20)",
        "g":        "#00e676",
        "g_d":      "rgba(0,230,118,0.10)",
        "g_b":      "rgba(0,230,118,0.25)",
        "r":        "#ff1744",
        "r_d":      "rgba(255,23,68,0.10)",
        "am":       "#ffab00",
        "t1":       "#eef0f6",
        "t2":       "#7b82a0",
        "t3":       "#444a63",
        "sb_bg":    "#0b0c10",
        "scroll":   "#242837",
        "hit_bg":   "#0a2e1a",
        "hit_fg":   "#00e676",
        "neg_bg":   "#2e0a0a",
        "neg_fg":   "#ff5252",
        "warn_bg":  "#2e1a00",
        "warn_fg":  "#ffab00",
    },
    "light": {
        "bg":       "#f4f5f9",
        "s1":       "#ffffff",
        "s2":       "#eef0f6",
        "s3":       "#e2e5ef",
        "b":        "rgba(0,0,0,0.08)",
        "bh":       "rgba(0,0,0,0.16)",
        "bs":       "rgba(0,0,0,0.05)",
        "acc":      "#0060df",
        "acc_d":    "rgba(0,96,223,0.10)",
        "acc_g":    "rgba(0,96,223,0.20)",
        "g":        "#00875a",
        "g_d":      "rgba(0,135,90,0.10)",
        "g_b":      "rgba(0,135,90,0.30)",
        "r":        "#c92a2a",
        "r_d":      "rgba(201,42,42,0.10)",
        "am":       "#e67700",
        "t1":       "#111318",
        "t2":       "#555d72",
        "t3":       "#8b93a7",
        "sb_bg":    "#eaecf3",
        "scroll":   "#c8ccd8",
        "hit_bg":   "#d3f9d8",
        "hit_fg":   "#087f5b",
        "neg_bg":   "#ffe3e3",
        "neg_fg":   "#c92a2a",
        "warn_bg":  "#fff3bf",
        "warn_fg":  "#e67700",
    },
}

T = THEMES[st.session_state.theme]

_css = (
    "@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap');"

    "html,body,[class*='css'],.stApp{background:" + T["bg"] + " !important;color:" + T["t1"] + " !important;font-family:'Space Grotesk',sans-serif !important;}"
    ".block-container{padding:1rem 2rem 3rem !important;max-width:1500px !important;}"
    "#MainMenu,footer,header{display:none !important;}"

    # Sidebar styling
    "[data-testid='stSidebar']{background-color:" + T["sb_bg"] + " !important;border-right:1px solid " + T["b"] + " !important;}"
    "[data-testid='stSidebar'] .block-container{padding:1.2rem 1rem !important;}"
    
    ".sb-brand{display:flex;align-items:center;gap:10px;margin-bottom:0.2rem;}"
    ".sb-brand-icon{font-size:1.3rem;color:" + T["acc"] + ";}"
    ".sb-brand-name{font-family:'JetBrains Mono',monospace;font-size:1.05rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:" + T["t1"] + ";}"
    ".sb-brand-tag{font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:" + T["t3"] + ";letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.9rem;}"

    ".sb-card{background:" + T["s1"] + ";border:1px solid " + T["b"] + ";border-radius:12px;padding:0.75rem 0.9rem;margin-bottom:0.9rem;}"
    ".sb-card-label{font-family:'JetBrains Mono',monospace;font-size:0.55rem;color:" + T["t3"] + ";letter-spacing:0.12em;text-transform:uppercase;margin-bottom:4px;}"
    ".sb-card-val{font-family:'JetBrains Mono',monospace;font-size:0.95rem;font-weight:600;color:" + T["t1"] + ";}"

    # Topbar styling
    ".topbar{display:flex;align-items:center;justify-content:space-between;padding:0.8rem 0;border-bottom:1px solid " + T["b"] + ";margin-bottom:1.4rem;position:sticky;top:0;z-index:90;background:" + T["bg"] + ";}"
    ".topbar-title{display:flex;align-items:baseline;gap:8px;font-family:'JetBrains Mono',monospace;font-size:1.15rem;font-weight:600;letter-spacing:0.04em;color:" + T["t1"] + ";}"
    ".topbar-subtitle{font-family:'Space Grotesk',sans-serif;font-size:0.75rem;color:" + T["t3"] + ";}"
    ".live-pill{display:flex;align-items:center;gap:6px;font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:" + T["g"] + ";background:" + T["g_d"] + ";border:1px solid " + T["g_b"] + ";padding:4px 10px;border-radius:100px;letter-spacing:0.06em;}"
    ".dot{width:5px;height:5px;border-radius:50%;background:" + T["g"] + ";animation:blink 2s infinite;}"
    "@keyframes blink{0%,100%{opacity:1}50%{opacity:.4}}"

    # Metric styling
    "[data-testid='stMetric']{background:" + T["s1"] + " !important;border:1px solid " + T["b"] + " !important;border-radius:12px !important;padding:0.9rem 1.1rem !important;transition:border-color .2s,transform .15s !important;}"
    "[data-testid='stMetric']:hover{border-color:" + T["bh"] + " !important;transform:translateY(-1px) !important;}"
    "[data-testid='stMetricLabel']{font-family:'JetBrains Mono',monospace !important;font-size:0.6rem !important;text-transform:uppercase !important;letter-spacing:0.14em !important;color:" + T["t3"] + " !important;}"
    "[data-testid='stMetricValue']{font-family:'JetBrains Mono',monospace !important;font-size:1.4rem !important;font-weight:600 !important;color:" + T["t1"] + " !important;letter-spacing:-0.02em !important;}"
    "[data-testid='stMetricDelta']{font-family:'JetBrains Mono',monospace !important;font-size:0.7rem !important;}"

    ".slabel{display:flex;align-items:center;gap:8px;font-family:'JetBrains Mono',monospace;font-size:0.58rem;font-weight:500;text-transform:uppercase;letter-spacing:0.2em;color:" + T["acc"] + ";margin-bottom:0.4rem;}"
    ".slabel::after{content:'';flex:1;height:1px;background:" + T["b"] + ";}"
    ".stitle{font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:600;letter-spacing:-0.02em;color:" + T["t1"] + ";margin-bottom:0.9rem;}"

    # Holdings card styling
    ".hcard{background:" + T["s1"] + ";border:1px solid " + T["b"] + ";border-radius:12px;padding:0.9rem 1.1rem;margin-bottom:8px;transition:border-color .18s,background .18s;position:relative;overflow:hidden;}"
    ".hcard-top{display:grid;grid-template-columns:3fr 2fr 2fr 2fr 2fr;align-items:center;gap:0;margin-bottom:0.55rem;}"
    ".hcard-bottom{display:grid;grid-template-columns:2fr 2fr 2fr 2fr;align-items:center;gap:0;padding-top:0.5rem;border-top:1px solid " + T["bs"] + ";}"
    ".hcard::before{content:'';position:absolute;left:0;top:0;width:3px;height:100%;border-radius:3px 0 0 3px;}"
    ".hcard.profit::before{background:" + T["g"] + ";}"
    ".hcard.loss::before{background:" + T["r"] + ";}"
    ".hcard:hover{border-color:" + T["bh"] + ";background:" + T["s2"] + ";}"

    ".hc-sym{font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:600;letter-spacing:0.04em;color:" + T["t1"] + ";}"
    ".hc-sub{font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:" + T["t3"] + ";margin-top:2px;letter-spacing:0.06em;}"
    ".hc-cell{display:flex;flex-direction:column;align-items:flex-end;gap:2px;}"
    ".hc-cell-label{font-family:'JetBrains Mono',monospace;font-size:0.55rem;color:" + T["t3"] + ";letter-spacing:.1em;text-transform:uppercase;}"
    ".hc-cell-val{font-family:'JetBrains Mono',monospace;font-size:0.82rem;font-weight:500;color:" + T["t2"] + ";}"
    ".hc-cell-val.hi{color:" + T["t1"] + ";font-weight:600;}"
    ".hc-cell-val.g{color:" + T["g"] + ";font-weight:600;}"
    ".hc-cell-val.r{color:" + T["r"] + ";font-weight:600;}"
    ".hc-cell-val.am{color:" + T["am"] + ";font-weight:600;}"

    "[data-testid='stDataFrame']{border:1px solid " + T["b"] + " !important;border-radius:12px !important;overflow:hidden !important;}"
    "[data-testid='stDataFrame']>div{background:" + T["s1"] + " !important;}"

    ".div{height:1px;background:" + T["b"] + ";margin:1.3rem 0;}"
    "::-webkit-scrollbar{width:4px;height:4px;}"
    "::-webkit-scrollbar-track{background:transparent;}"
    "::-webkit-scrollbar-thumb{background:" + T["scroll"] + ";border-radius:3px;}"

    # ── Sidebar collapse/expand arrow ─────────────────────────────────────────
    # Collapse (‹) button — inside the open sidebar.
    "[data-testid='stSidebarCollapseButton']{"
    "display:flex !important;visibility:visible !important;opacity:1 !important;}"
    "[data-testid='stSidebarCollapseButton'] button{"
    "background:" + T["s3"] + " !important;"
    "border:1px solid " + T["bh"] + " !important;"
    "border-radius:8px !important;"
    "width:32px !important;height:32px !important;"
    "transition:background .2s,border-color .2s !important;}"
    "[data-testid='stSidebarCollapseButton'] button svg{"
    "fill:" + T["t1"] + " !important;stroke:" + T["t1"] + " !important;color:" + T["t1"] + " !important;}"
    "[data-testid='stSidebarCollapseButton'] button:hover{"
    "background:" + T["acc_d"] + " !important;"
    "border-color:" + T["acc"] + " !important;}"
    "[data-testid='stSidebarCollapseButton'] button:hover svg{"
    "fill:" + T["acc"] + " !important;stroke:" + T["acc"] + " !important;color:" + T["acc"] + " !important;}"

    # Open (›) button — floats in main area when sidebar is collapsed.
    "[data-testid='stSidebarOpenButton']{"
    "display:flex !important;visibility:visible !important;opacity:1 !important;"
    "position:fixed !important;top:12px !important;left:12px !important;z-index:9999 !important;}"
    "[data-testid='stSidebarOpenButton'] button{"
    "background:" + T["acc"] + " !important;"
    "border:none !important;"
    "border-radius:10px !important;"
    "width:38px !important;height:38px !important;"
    "box-shadow:0 2px 16px " + T["acc_g"] + " !important;"
    "transition:transform .15s,box-shadow .2s !important;}"
    "[data-testid='stSidebarOpenButton'] button svg{"
    "fill:" + T["bg"] + " !important;stroke:" + T["bg"] + " !important;color:" + T["bg"] + " !important;}"
    "[data-testid='stSidebarOpenButton'] button:hover{"
    "transform:scale(1.08) !important;"
    "box-shadow:0 4px 24px " + T["acc_g"] + " !important;}"

    # ── Sidebar text / widget colours (light theme needs explicit overrides) ──
    "[data-testid='stSidebar'] *{"
    "color:" + T["t1"] + " !important;}"
    "[data-testid='stSidebar'] label,"
    "[data-testid='stSidebar'] .stRadio label,"
    "[data-testid='stSidebar'] [data-testid='stWidgetLabel'],"
    "[data-testid='stSidebar'] p,"
    "[data-testid='stSidebar'] span{"
    "color:" + T["t1"] + " !important;}"
    # Radio option text
    "[data-testid='stSidebar'] [data-testid='stRadio'] label span{"
    "color:" + T["t1"] + " !important;}"
    # Selected radio item
    "[data-testid='stSidebar'] [data-testid='stRadio'] [aria-checked='true'] span{"
    "color:" + T["acc"] + " !important;font-weight:600 !important;}"
    # Expander header text
    "[data-testid='stSidebar'] [data-testid='stExpander'] summary *{"
    "color:" + T["t1"] + " !important;}"
    # Number input / select label
    "[data-testid='stSidebar'] .stNumberInput label,"
    "[data-testid='stSidebar'] .stSelectbox label,"
    "[data-testid='stSidebar'] .stCheckbox label{"
    "color:" + T["t2"] + " !important;}"
    # Caption / helper text
    "[data-testid='stSidebar'] .stCaption *{"
    "color:" + T["t3"] + " !important;}"
)

st.markdown("<style>" + _css + "</style>", unsafe_allow_html=True)

def _sl(label):
    st.markdown(f'<div class="slabel">{label}</div>', unsafe_allow_html=True)

def _st(title):
    st.markdown(f'<div class="stitle">{title}</div>', unsafe_allow_html=True)

def _div():
    st.markdown('<div class="div"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED DATA FETCHING LAYER (@st.cache_data)
# ══════════════════════════════════════════════════════════════════════════════

def _clean_slug(repo_str):
    slug = repo_str.strip().rstrip("/")
    for prefix in ["https://github.com/", "http://github.com/"]:
        if slug.startswith(prefix):
            slug = slug[len(prefix):]
    return slug


@st.cache_data(ttl=600, show_spinner=False)
def load_watchlist_cached(file_path, token="", repo="", fallback=None):
    """
    Loads watchlist tickers with caching (TTL 10m).
    Tries GitHub API first if credentials exist, otherwise falls back to local disk.
    """
    tickers = []
    err = None
    source = "fallback"

    # 1. Try GitHub API if credentials configured
    if token and repo:
        try:
            repo_slug = _clean_slug(repo)
            url = f"https://api.github.com/repos/{repo_slug}/contents/{file_path}"
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }
            r = requests.get(url, headers=headers, timeout=8)
            if r.status_code == 200:
                raw = base64.b64decode(r.json()["content"]).decode("utf-8")
                tickers = [t.strip().upper() for t in raw.replace("\n", ",").split(",") if t.strip()]
                return tickers, None, "github"
            else:
                err = f"GitHub API {r.status_code}: {r.json().get('message', 'Failed')}"
        except Exception as e:
            err = f"GitHub connection error: {str(e)}"

    # 2. Try local file system
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            tickers = [t.strip().upper() for t in content.replace("\n", ",").split(",") if t.strip()]
            return tickers, None, "local"
        except Exception as e:
            err = f"Local file read error: {str(e)}"

    # 3. Use fallback list
    if fallback:
        return list(fallback), err, "default_fallback"

    return [], err, "empty"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_tradebook_cached(token="", repo="", filename=CSV_FILENAME):
    """
    Loads tradebook DataFrame with caching (TTL 5m).
    """
    if token and repo:
        try:
            repo_slug = _clean_slug(repo)
            url = f"https://api.github.com/repos/{repo_slug}/contents/{filename}"
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                raw_csv = base64.b64decode(r.json()["content"]).decode("utf-8")
                return pd.read_csv(io.StringIO(raw_csv)), None
            elif r.status_code != 404:
                return pd.DataFrame(), f"GitHub HTTP {r.status_code}"
        except Exception as e:
            pass

    # Local fallback
    if os.path.exists(filename):
        try:
            return pd.read_csv(filename), None
        except Exception as e:
            return pd.DataFrame(), f"Local CSV read error: {e}"

    return pd.DataFrame(), "Tradebook file not found"


@st.cache_data(ttl=120, show_spinner=False)
def fetch_manual_trades_cached(token="", repo="", filename=MANUAL_CSV_FILENAME):
    """
    Loads manual trades DataFrame and git SHA with caching (TTL 2m).
    """
    if token and repo:
        try:
            repo_slug = _clean_slug(repo)
            url = f"https://api.github.com/repos/{repo_slug}/contents/{filename}"
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                data = r.json()
                sha = data.get("sha")
                raw = base64.b64decode(data.get("content", "")).decode("utf-8")
                if not raw.strip():
                    return pd.DataFrame(), sha, None
                df = pd.read_csv(io.StringIO(raw))
                return df, sha, None
        except Exception as e:
            pass

    # Local fallback
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            return df, "local", None
        except Exception as e:
            return pd.DataFrame(), None, f"Local manual read error: {e}"

    return pd.DataFrame(), None, None


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def compute_indicators_cached(symbols_tuple):
    """
    Downloads 5 years of daily candles and calculates 200 DMA, Wilder RSI,
    Bollinger Bands, ATH, 52W High with long TTL caching.
    """
    master_frames = {}
    indicator_data = {}
    if not symbols_tuple:
        return master_frames, indicator_data

    end = datetime.today()
    start = end - timedelta(days=5 * 365)
    tickers = " ".join(f"{s}.NS" for s in symbols_tuple)

    try:
        batch = yf.download(
            tickers, start=start, end=end,
            progress=False, group_by="ticker", auto_adjust=True
        )
    except Exception:
        batch = pd.DataFrame()

    for sym in symbols_tuple:
        try:
            if len(symbols_tuple) == 1:
                df = batch.copy()
            else:
                key = sym + ".NS"
                if key not in batch.columns.get_level_values(0):
                    raise ValueError(f"{sym} not in download batch")
                df = batch[key].copy()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Close"])
            if df.empty or len(df) < 2:
                raise ValueError("Insufficient candles")

            df["RSI"] = calculate_wilder_rsi(df["Close"], 14)
            df["200_DMA"] = df["Close"].rolling(200, min_periods=20).mean()
            bm = df["Close"].rolling(20, min_periods=5).mean()
            bs = df["Close"].rolling(20, min_periods=5).std().fillna(0)
            df["BB_LOW"] = bm - 2 * bs
            master_frames[sym] = df

            lat = df.iloc[-1]
            prev = df.iloc[-2]
            ltp = float(lat["Close"])
            pc = float(prev["Close"])
            ath = float(df["Close"].max())
            yr = df.tail(252)
            w52 = float(yr["High"].max()) if "High" in yr.columns else ath

            indicator_data[sym] = {
                "LTP": ltp,
                "Day_Change_Pct": ((ltp - pc) / pc * 100) if pc else 0.0,
                "RSI": float(lat["RSI"]) if not np.isnan(lat["RSI"]) else 50.0,
                "200_DMA": float(lat["200_DMA"]) if not np.isnan(lat["200_DMA"]) else ltp,
                "BB_LOW": float(lat["BB_LOW"]) if not np.isnan(lat["BB_LOW"]) else ltp,
                "ATH": ath,
                "ATH_Dist_Pct": ((ltp - ath) / ath * 100) if ath else 0.0,
                "52W_High": w52,
                "vs_52W_Pct": ((ltp - w52) / w52 * 100) if w52 else 0.0,
            }
        except Exception:
            # Fallback to single quote or fast_info
            try:
                t = yf.Ticker(f"{sym}.NS")
                lp = float(t.fast_info.get("lastPrice", 0) or 0)
                if lp > 0:
                    indicator_data[sym] = {
                        "LTP": lp, "Day_Change_Pct": 0.0, "RSI": 50.0,
                        "200_DMA": lp, "BB_LOW": lp, "ATH": lp,
                        "ATH_Dist_Pct": 0.0, "52W_High": lp, "vs_52W_Pct": 0.0,
                    }
            except Exception:
                pass

    return master_frames, indicator_data


@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_ltp_batch_cached(symbols_tuple):
    """
    Fetches real-time intraday quotes (TTL 60s).
    """
    live = {}
    if not symbols_tuple:
        return live

    tickers = " ".join(f"{s}.NS" for s in symbols_tuple)
    try:
        batch = yf.download(
            tickers, period="2d", interval="1m",
            progress=False, group_by="ticker", auto_adjust=True
        )
        today_ts = pd.Timestamp.now(tz="Asia/Kolkata").normalize()
        for sym in symbols_tuple:
            try:
                if len(symbols_tuple) == 1:
                    df = batch.copy()
                else:
                    key = sym + ".NS"
                    if key not in batch.columns.get_level_values(0):
                        continue
                    df = batch[key].copy()

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna(subset=["Close"])
                if df.empty:
                    continue

                ltp = float(df["Close"].iloc[-1])
                prev_df = df[df.index.normalize() < today_ts]
                pc = float(prev_df["Close"].iloc[-1]) if not prev_df.empty else ltp
                day_chg = ((ltp - pc) / pc * 100) if pc else 0.0
                live[sym] = {"ltp": ltp, "day_chg_pct": day_chg}
            except Exception:
                pass
    except Exception:
        pass

    return live


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def compute_swing_analysis_cached(symbols_tuple, threshold):
    """
    Computes swing floor and peak pivots with caching.
    """
    results = []
    if not symbols_tuple:
        return results

    # Attempt to load 3 years for pivots
    end = datetime.today()
    start = end - timedelta(days=3 * 365)
    tickers = " ".join(f"{s}.NS" for s in symbols_tuple)

    try:
        batch = yf.download(tickers, start=start, end=end,
                            progress=False, group_by="ticker", auto_adjust=True)
    except Exception:
        batch = pd.DataFrame()

    for sym in symbols_tuple:
        try:
            if len(symbols_tuple) == 1:
                df = batch.copy()
            else:
                key = sym + ".NS"
                if key not in batch.columns.get_level_values(0):
                    continue
                df = batch[key].copy()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Close"])
            if df.empty or len(df) < 20:
                continue

            pivot_res = detect_swing_pivots(df, threshold_pct=threshold)
            if pivot_res:
                pivot_res["sym"] = sym
                results.append(pivot_res)
        except Exception:
            pass

    return results


def commit_file_to_github(content_str, filename, sha=None, message="Update"):
    """
    Commits text or base64 file to GitHub repository.
    """
    if not GITHUB_TOKEN or not GITHUB_REPO:
        # Fallback to local save
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content_str)
            return True, None, "saved_locally"
        except Exception as e:
            return False, f"Local write error: {e}", "error"

    try:
        repo_slug = _clean_slug(GITHUB_REPO)
        url = f"https://api.github.com/repos/{repo_slug}/contents/{filename}"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        b64 = base64.b64encode(content_str.encode("utf-8")).decode("utf-8")
        payload = {"message": message, "content": b64}
        if sha and sha != "local":
            payload["sha"] = sha
        r = requests.put(url, headers=headers, json=payload, timeout=15)
        if r.status_code in (200, 201):
            return True, None, "github"
        return False, f"GitHub API {r.status_code}: {r.json().get('message','')}", "github_error"
    except Exception as e:
        return False, str(e), "exception"


# ══════════════════════════════════════════════════════════════════════════════
# MARKET STATUS & AUTO REFRESH
# ══════════════════════════════════════════════════════════════════════════════
now_ist = get_ist_now()
market_open = is_market_open(now_ist)

if market_open and st.session_state.auto_refresh_enabled:
    if AUTOREFRESH_AVAILABLE:
        st_autorefresh(interval=st.session_state.refresh_interval * 1000, key="mkt_refresh")
    
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    elapsed = time.time() - st.session_state.last_refresh
    if elapsed >= st.session_state.refresh_interval:
        fetch_live_ltp_batch_cached.clear()
        st.session_state.last_refresh = time.time()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# INITIALIZE DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading watchlists & tradebooks…"):
    # Load watchlists
    stk_t, stk_err, stk_src = load_watchlist_cached(STOCK_FILE_PATH, GITHUB_TOKEN, GITHUB_REPO, [])
    etf_t, etf_err, etf_src = load_watchlist_cached(ETF_FILE_PATH, GITHUB_TOKEN, GITHUB_REPO, DEFAULT_ETF_LIST)
    n50_t, n50_err, n50_src = load_watchlist_cached(NIFTY50_FILE_PATH, GITHUB_TOKEN, GITHUB_REPO, [])
    nn50_t, nn50_err, nn50_src = load_watchlist_cached(NIFTYNEXT50_FILE_PATH, GITHUB_TOKEN, GITHUB_REPO, [])

    st.session_state.Stock_watchlist = stk_t
    st.session_state.ETF_watchlist = etf_t
    st.session_state.Nifty50_watchlist = list(dict.fromkeys(n50_t))
    st.session_state.NiftyNext50_watchlist = list(dict.fromkeys(nn50_t))

    # Fetch tradebook
    raw_tb, tb_err = fetch_tradebook_cached(GITHUB_TOKEN, GITHUB_REPO, CSV_FILENAME)
    tradebook_df = parse_tradebook(raw_tb)

    # Fetch manual trades
    raw_man, man_sha, man_err = fetch_manual_trades_cached(GITHUB_TOKEN, GITHUB_REPO, MANUAL_CSV_FILENAME)
    manual_df = parse_tradebook(raw_man)
    st.session_state.manual_trades_sha = man_sha
    st.session_state.manual_trades_df = manual_df

    # Combine trades
    all_trades = tradebook_df.copy()
    if not manual_df.empty:
        all_trades = pd.concat([all_trades, manual_df], ignore_index=True)

    # Process LOFO engine
    closed_trades, open_data = process_lofo(all_trades)

holding_symbols = [sym for sym, d in open_data.items() if d["remaining_qty"] > 0]
filtered_stock_pool = [t for t in st.session_state.Stock_watchlist if t not in holding_symbols]
filtered_etf_pool = [t for t in st.session_state.ETF_watchlist if t not in holding_symbols]
filtered_nifty50_pool = [t for t in st.session_state.Nifty50_watchlist if t not in holding_symbols]
filtered_niftynext50_pool = [t for t in st.session_state.NiftyNext50_watchlist if t not in holding_symbols]

total_download_pool = list(set(holding_symbols + filtered_stock_pool + filtered_etf_pool + filtered_nifty50_pool + filtered_niftynext50_pool))
symbols_key = tuple(sorted(total_download_pool))

# Compute Indicators
with st.spinner(f"Updating market indicators for {len(symbols_key)} instruments…"):
    history_frames, indicators = compute_indicators_cached(symbols_key)

summary_feeds = {sym: dict(vals) for sym, vals in indicators.items()}

# Apply live prices
if market_open or "closed_prices" not in st.session_state:
    with st.spinner("Fetching live intraday quotes…"):
        live_prices = fetch_live_ltp_batch_cached(symbols_key)
        apply_live_prices(summary_feeds, live_prices)
        if not market_open:
            st.session_state.closed_prices = live_prices
else:
    apply_live_prices(summary_feeds, st.session_state.closed_prices)

today_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

# Calculate high-level portfolio metrics
blocks_map = split_lots_into_blocks(open_data, BLOCK_SIZE)
tot_invested = 0.0
tot_value = 0.0
global_cfs = []
global_dates = []

for _, t in all_trades.iterrows():
    s = -1 if t['Trade Type'] == 'buy' else 1
    global_cfs.append(s * t['Quantity'] * t['Price'])
    global_dates.append(t['Trade Date'])

holding_meta = {}
all_profitable_tranches = []
closed_by_sym = {}
for ct in closed_trades:
    closed_by_sym.setdefault(ct['Symbol'], []).append(ct)

for sym, d in open_data.items():
    if d['remaining_qty'] == 0:
        continue
    ltp = summary_feeds.get(sym, {}).get('LTP', d['avg_price'])
    day_chg = summary_feeds.get(sym, {}).get('Day_Change_Pct', 0.0)
    cur_val = d['remaining_qty'] * ltp
    pnl_amt = cur_val - d['invested_amount']
    pnl_pct = (pnl_amt / d['invested_amount'] * 100) if d['invested_amount'] else 0.0
    tot_invested += d['invested_amount']
    tot_value += cur_val

    global_cfs.append(cur_val)
    global_dates.append(today_dt)

    open_cfs = [-l['remaining_qty'] * l['price'] for l in d['lots']]
    open_dates = [l['date'] for l in d['lots']]
    open_cfs.append(cur_val)
    open_dates.append(today_dt)
    xirr_open = calculate_xirr(open_cfs, open_dates)

    sym_closed = closed_by_sym.get(sym, [])
    if sym_closed:
        cl_cfs = []
        cl_dates = []
        for ct in sym_closed:
            cl_cfs.append(-ct['Buy Value'])
            cl_dates.append(ct['Buy Date'])
            cl_cfs.append(ct['Sell Value'])
            cl_dates.append(ct['Sell Date'])
        xirr_closed = calculate_xirr(cl_cfs, cl_dates)
    else:
        xirr_closed = None

    last_close = d['avg_price']
    last_close_date = today_dt
    if sym in history_frames and not history_frames[sym].empty:
        hdf_sym = history_frames[sym]
        last_close = float(hdf_sym['Close'].iloc[-1])
        last_close_date = hdf_sym.index[-1].to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)

    comb_cfs = [-l['remaining_qty'] * l['price'] for l in d['lots']]
    comb_dates = [l['date'] for l in d['lots']]
    for ct in sym_closed:
        comb_cfs.append(-ct['Buy Value'])
        comb_dates.append(ct['Buy Date'])
        comb_cfs.append(ct['Sell Value'])
        comb_dates.append(ct['Sell Date'])
    if d['remaining_qty'] > 0:
        comb_cfs.append(d['remaining_qty'] * last_close)
        comb_dates.append(last_close_date)
    xirr_combined = calculate_xirr(comb_cfs, comb_dates)

    realized_pnl = sum(ct['Profit'] for ct in sym_closed)

    holding_meta[sym] = {
        'ltp': ltp, 'day_chg': day_chg, 'cur_val': cur_val,
        'pnl_amt': pnl_amt, 'pnl_pct': pnl_pct, 'invested': d['invested_amount'],
        'xirr_open': xirr_open, 'xirr_closed': xirr_closed,
        'xirr_combined': xirr_combined, 'realized_pnl': realized_pnl,
    }

    # Profitable tranches calculation
    for b in blocks_map.get(sym, []):
        bdf = pd.DataFrame(b['lots'])
        b_qty = b['qty']
        b_avg = bdf['price'].sum() / b_qty
        b_inv = bdf['price'].sum()
        b_cur = b_qty * ltp
        b_pnl = b_cur - b_inv
        b_pct = b_pnl / b_inv * 100 if b_inv else 0
        bxirr_tr = calculate_xirr(
            [-lot['price'] for lot in b['lots']] + [b_cur],
            [lot['date'] for lot in b['lots']] + [today_dt]
        )
        w_date_tr = get_xirr_weighted_buy_date(b['lots'], b_cur, bxirr_tr, today_dt)
        h_close = b_avg
        h_close_dt = ""
        if sym in history_frames:
            hdf = history_frames[sym]
            post = hdf[hdf.index >= pd.to_datetime(w_date_tr).tz_localize(None)]
            if not post.empty and 'Close' in post.columns:
                peak_idx = post['Close'].idxmax()
                h_close = float(post['Close'].loc[peak_idx])
                h_close_dt = pd.to_datetime(peak_idx).strftime('%d %b %y')
        fall_pct = ((ltp - h_close) / h_close * 100) if h_close > 0 else 0.0
        if b_pct >= 0.0 and sym != "LIQUIDCASE":
            all_profitable_tranches.append({
                'Symbol': sym,
                'Block': b['block_name'],
                'Qty': b_qty,
                'Avg Cost': b_avg,
                'LTP': ltp,
                'Peak Close': h_close,
                'Peak Date': h_close_dt,
                'Return %': b_pct,
                'Gain ₹': b_pnl,
                'Block XIRR %': bxirr_tr,
                'Fall from Peak %': fall_pct,
            })

net_pnl = tot_value - tot_invested
net_pct = (net_pnl / tot_invested * 100) if tot_invested > 0 else 0.0
acct_xirr = calculate_xirr(global_cfs, global_dates)
tot_booked_pnl = sum(ct['Profit'] for ct in closed_trades) if closed_trades else 0.0

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION & CONTROLS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div class='sb-brand'>"
        "<div class='sb-brand-icon'>▣</div>"
        "<div class='sb-brand-name'>LOFO Terminal</div>"
        "</div>"
        "<div class='sb-brand-tag'>NSE Portfolio Intelligence</div>",
        unsafe_allow_html=True
    )

    # Market status card
    status_color = T["g"] if market_open else T["am"]
    status_label = "● Live Market" if market_open else "⏸ Market Closed"
    time_str = now_ist.strftime("%H:%M:%S IST")
    date_str = now_ist.strftime("%d %b %Y")

    st.markdown(
        f"<div class='sb-card'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
        f"<span style='color:{status_color};font-weight:600;font-size:0.75rem;font-family:JetBrains Mono;'>{status_label}</span>"
        f"<span style='color:{T['t3']};font-size:0.65rem;font-family:JetBrains Mono;'>{time_str}</span>"
        f"</div>"
        f"<div style='color:{T['t2']};font-size:0.6rem;font-family:JetBrains Mono;margin-top:2px;'>Trading Hours: 09:15 - 15:30 IST</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    _sl("NAVIGATION")
    nav_options = [
        "📊 Portfolio Holdings",
        "🎯 Watchlist Scanner",
        "📈 Technical Charts",
        "📜 Closed Trades & P&L",
        "⚡ Swing Radar",
        "📥 Upload & Manual Trades",
        "💾 Data Export"
    ]
    selected_nav = st.radio("Go to", nav_options, label_visibility="collapsed")

    _div()

    # Quick Summary Card
    _sl("PORTFOLIO GLANCE")
    pnl_sign = "+" if net_pnl >= 0 else ""
    pnl_col = T["g"] if net_pnl >= 0 else T["r"]
    st.markdown(
        f"<div class='sb-card'>"
        f"<div class='sb-card-label'>Invested Capital</div>"
        f"<div class='sb-card-val'>₹{tot_invested:,.0f}</div>"
        f"<div style='height:6px;'></div>"
        f"<div class='sb-card-label'>Current Valuation</div>"
        f"<div class='sb-card-val'>₹{tot_value:,.0f}</div>"
        f"<div style='height:6px;'></div>"
        f"<div class='sb-card-label'>Unrealized P&L</div>"
        f"<div style='font-family:JetBrains Mono;font-size:0.95rem;font-weight:600;color:{pnl_col};'>"
        f"{pnl_sign}₹{net_pnl:,.0f} ({pnl_sign}{net_pct:.2f}%)</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Terminal Settings
    with st.expander("⚙ Terminal Settings", expanded=False):
        # Theme toggle
        _tlabel = '☀ Switch to Light Mode' if st.session_state.theme == 'dark' else '🌙 Switch to Dark Mode'
        if st.button(_tlabel, use_container_width=True):
            st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
            st.rerun()

        # Block Size setting
        new_bs = st.number_input("Capital Block Size (₹)", min_value=1000, max_value=500000, value=st.session_state.block_size, step=1000)
        if new_bs != st.session_state.block_size:
            st.session_state.block_size = new_bs
            st.rerun()

        # Auto refresh toggle
        auto_ref = st.checkbox("Enable Auto-Refresh", value=st.session_state.auto_refresh_enabled)
        st.session_state.auto_refresh_enabled = auto_ref

        if auto_ref:
            interval = st.selectbox("Interval (sec)", [60, 120, 180, 300], index=2)
            st.session_state.refresh_interval = interval

        # Clear Cache / Force Refresh Button
        if st.button("🔄 Force Refresh All Caches", use_container_width=True):
            st.cache_data.clear()
            st.session_state.pop("closed_prices", None)
            st.rerun()

        # Connection status
        gh_connected = bool(GITHUB_TOKEN and GITHUB_REPO)
        gh_status = "🟢 GitHub Sync Active" if gh_connected else "📁 Local Storage Mode"
        st.caption(f"Status: **{gh_status}**")


# ══════════════════════════════════════════════════════════════════════════════
# TOP HEADER BAR
# ══════════════════════════════════════════════════════════════════════════════
topbar_html = (
    "<div class='topbar'>"
    "<div class='topbar-title'>"
    f"<div>{selected_nav}</div>"
    "</div>"
    "<div style='display:flex;align-items:center;gap:12px;'>"
    f"<span style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:{T['t3']};'>{date_str} {time_str}</span>"
    + (f"<div class='live-pill'><div class='dot'></div>Live · {time_str}</div>" if market_open else f"<div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:{T['t3']};background:{T['s2']};border:1px solid {T['b']};padding:4px 10px;border-radius:100px;'>⏸ Market Closed</div>") +
    "</div></div>"
)
st.markdown(topbar_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 1: PORTFOLIO HOLDINGS
# ══════════════════════════════════════════════════════════════════════════════
if selected_nav == "📊 Portfolio Holdings":
    if not open_data or not any(d['remaining_qty'] > 0 for d in open_data.values()):
        st.info("No active holdings found in tradebook. Upload a tradebook or add manual trades to begin.")
    else:
        day_change_amt = 0.0
        for sym in holding_symbols:
            d = open_data[sym]
            m = holding_meta.get(sym, {})
            ltp = m.get("ltp", d["avg_price"])
            day_chg_pct = m.get("day_chg", 0.0)
            if day_chg_pct != 0:
                prev_close = ltp / (1 + day_chg_pct / 100)
                day_change_amt += (ltp - prev_close) * d["remaining_qty"]
        day_change_pct_port = (day_change_amt / tot_value * 100) if tot_value else 0.0

        _sl("ACCOUNT OVERVIEW")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Deployed", f"₹{tot_invested:,.0f}")
        c2.metric("Market Value", f"₹{tot_value:,.0f}")
        pnl_sign = "+" if net_pnl >= 0 else ""
        day_sign = "+" if day_change_amt >= 0 else ""
        c3.metric("Unrealised P&L", f"₹{net_pnl:,.0f}", f"{pnl_sign}{net_pct:.2f}%")
        c4.metric("Today's Change", f"₹{day_change_amt:,.0f}", f"{day_sign}{day_change_pct_port:.2f}%")
        c5.metric("Portfolio XIRR", f"{acct_xirr:.2f}%")

        _div()

        _sl("OPERATIONAL RADAR")
        _st("Profitable Tranche Intelligence")
        if all_profitable_tranches:
            def _cr(v):
                try:
                    n = float(str(v).replace('%','').replace('₹','').replace(',',''))
                    return 'color:' + T['g'] + ';font-weight:600' if n > 0 else (
                           'color:' + T['r'] + ';font-weight:600' if n < 0 else '')
                except Exception:
                    return ''

            def _cf(v):
                try:
                    n = float(str(v).replace('%','').replace('₹','').replace(',',''))
                    if n < -5: return 'color:' + T['r'] + ';font-weight:600'
                    if n <  0: return 'color:' + T['am'] + ';font-weight:500'
                except Exception:
                    return ''

            def _cx(v):
                try:
                    n = float(str(v).replace('%','').replace('₹','').replace(',',''))
                    if n > 0: return 'color:' + T['g'] + ';font-weight:600'
                    if n < 0: return 'color:' + T['r'] + ';font-weight:600'
                except Exception:
                    return ''

            _tr_fmt = {
                'Avg Cost':         '₹{:.2f}',
                'LTP':              '₹{:.2f}',
                'Peak Close':       '₹{:.2f}',
                'Return %':         '{:.2f}%',
                'Gain ₹':           '₹{:.2f}',
                'Block XIRR %':     '{:.2f}%',
                'Fall from Peak %': '{:.2f}%',
            }
            _tr_col_cfg = {
                'Peak Date':    st.column_config.TextColumn('Peak Date',  width='small'),
                'Block XIRR %': st.column_config.NumberColumn('Block XIRR %'),
            }

            def _render_tranche_table(df):
                st.dataframe(
                    df.style
                      .format(_tr_fmt)
                      .map(_cr, subset=['Return %'])
                      .map(_cx, subset=['Block XIRR %'])
                      .map(_cf, subset=['Fall from Peak %']),
                    use_container_width=True, hide_index=True,
                    column_config=_tr_col_cfg
                )

            df_tr = pd.DataFrame(all_profitable_tranches)
            df_high = df_tr[df_tr['Gain ₹'] >  150].reset_index(drop=True)
            df_low  = df_tr[df_tr['Gain ₹'] <= 150].reset_index(drop=True)

            _sl("TRANCHES · GAIN > ₹150")
            if not df_high.empty:
                _render_tranche_table(df_high)
            else:
                st.info("No tranches with gain above ₹150.")

            _div()
            _sl("TRANCHES · GAIN ≤ ₹150")
            if not df_low.empty:
                _render_tranche_table(df_low)
            else:
                st.info("No tranches with gain ₹150 or below.")
        else:
            st.info("No tranches in positive territory yet.")

        _div()

        _sl("POSITIONS")
        _st("Active Holdings Ledger")

        ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 3])
        with ctrl1:
            sort_by = st.selectbox(
                "Sort by",
                options=[
                    "Alphabetically",
                    "Day Change %",
                    "P&L Abs ₹",
                    "P&L Pct %",
                    "Invested ₹",
                    "Realized P&L ₹",
                    "Open XIRR %",
                    "Closed XIRR %",
                    "Combined XIRR %",
                ],
                label_visibility="visible"
            )
        with ctrl2:
            sort_dir = st.selectbox("Order", ["Descending", "Ascending"], label_visibility="visible")
        with ctrl3:
            filter_q = st.text_input("Filter symbol", placeholder="e.g. HDFC", label_visibility="visible").upper().strip()

        asc = sort_dir == "Ascending"

        sort_key_map = {
            "Alphabetically":   lambda s: s,
            "Day Change %":     lambda s: holding_meta[s]['day_chg'],
            "P&L Abs ₹":       lambda s: holding_meta[s]['pnl_amt'],
            "P&L Pct %":       lambda s: holding_meta[s]['pnl_pct'],
            "Invested ₹":      lambda s: holding_meta[s]['invested'],
            "Realized P&L ₹":  lambda s: holding_meta[s]['realized_pnl'],
            "Open XIRR %":     lambda s: holding_meta[s]['xirr_open'],
            "Closed XIRR %":   lambda s: holding_meta[s]['xirr_closed'] or 0,
            "Combined XIRR %": lambda s: holding_meta[s]['xirr_combined'],
        }
        keyfn = sort_key_map[sort_by]
        filtered_holdings = [s for s in holding_symbols if filter_q in s] if filter_q else holding_symbols[:]
        filtered_holdings.sort(key=keyfn, reverse=not asc)

        if not filtered_holdings:
            st.info("No holdings match the filter.")

        for sym in filtered_holdings:
            d   = open_data[sym]
            m   = holding_meta[sym]
            ltp = m['ltp']
            cur_val  = m['cur_val']
            pnl_amt  = m['pnl_amt']
            pnl_pct  = m['pnl_pct']
            day_chg  = m['day_chg']

            card_cls  = "profit" if pnl_amt>=0 else "loss"
            pnl_cls   = "g" if pnl_amt>=0 else "r"
            chg_cls   = "g" if day_chg>=0 else "r"
            pnl_sign  = "+" if pnl_amt>=0 else ""
            chg_arrow = "▲" if day_chg>=0 else "▼"

            xirr_open     = m['xirr_open']
            xirr_closed   = m['xirr_closed']
            xirr_combined = m['xirr_combined']
            realized_pnl  = m['realized_pnl']

            xirr_closed_str = f"{xirr_closed:.2f}%" if xirr_closed is not None else "—"
            xirr_closed_cls = ("g" if xirr_closed and xirr_closed>=0 else "r") if xirr_closed is not None else "hi"

            rpnl_cls  = "g" if realized_pnl>=0 else "r"
            rpnl_sign = "+" if realized_pnl>=0 else ""
            xo_cls    = "g" if xirr_open>=0 else "r"
            xc_cls    = xirr_closed_cls
            xcb_cls   = "g" if xirr_combined>=0 else "r"

            st.markdown(
                "<div class='hcard " + card_cls + "'>"
                "<div class='hcard-top'>"
                "<div>"
                "<div class='hc-sym'>" + sym + "</div>"
                "<div class='hc-sub'>" + str(d['remaining_qty']) + " units · avg ₹" + f"{d['avg_price']:.2f}" + "</div>"
                "</div>"
                "<div class='hc-cell'>"
                "<div class='hc-cell-label'>LTP</div>"
                "<div class='hc-cell-val hi'>₹" + f"{ltp:.2f}" + "</div>"
                "</div>"
                "<div class='hc-cell'>"
                "<div class='hc-cell-label'>Day Chg</div>"
                "<div class='hc-cell-val " + chg_cls + "'>" + chg_arrow + " " + f"{abs(day_chg):.2f}" + "%</div>"
                "</div>"
                "<div class='hc-cell'>"
                "<div class='hc-cell-label'>P&amp;L ₹</div>"
                "<div class='hc-cell-val " + pnl_cls + "'>" + pnl_sign + "₹" + f"{pnl_amt:,.0f}" + "</div>"
                "</div>"
                "<div class='hc-cell'>"
                "<div class='hc-cell-label'>P&amp;L %</div>"
                "<div class='hc-cell-val " + pnl_cls + "'>" + pnl_sign + f"{pnl_pct:.2f}" + "%</div>"
                "</div>"
                "</div>"
                "<div class='hcard-bottom'>"
                "<div class='hc-cell' style='align-items:flex-start;'>"
                "<div class='hc-cell-label'>Realized P&amp;L</div>"
                "<div class='hc-cell-val " + rpnl_cls + "'>" + rpnl_sign + "₹" + f"{realized_pnl:,.0f}" + "</div>"
                "</div>"
                "<div class='hc-cell'>"
                "<div class='hc-cell-label'>Open XIRR</div>"
                "<div class='hc-cell-val " + xo_cls + "'>" + f"{xirr_open:.2f}" + "%</div>"
                "</div>"
                "<div class='hc-cell'>"
                "<div class='hc-cell-label'>Closed XIRR</div>"
                "<div class='hc-cell-val " + xc_cls + "'>" + xirr_closed_str + "</div>"
                "</div>"
                "<div class='hc-cell'>"
                "<div class='hc-cell-label'>Combined XIRR</div>"
                "<div class='hc-cell-val " + xcb_cls + "'>" + f"{xirr_combined:.2f}" + "%</div>"
                "</div>"
                "</div>"
                "</div>",
                unsafe_allow_html=True
            )

            with st.expander(f"Block detail — {sym}"):
                b_list = blocks_map.get(sym, [])
                blk_rows = []
                for b in b_list:
                    bdf = pd.DataFrame(b['lots'])
                    b_qty = b['qty']
                    b_avg = bdf['price'].sum() / b_qty
                    b_inv = bdf['price'].sum()
                    b_cur = b_qty * ltp
                    b_pnl = b_cur - b_inv
                    bxirr = calculate_xirr([-lot['price'] for lot in b['lots']] + [b_cur],
                                           [lot['date'] for lot in b['lots']] + [today_dt])
                    w_date = get_xirr_weighted_buy_date(b['lots'], b_cur, bxirr, today_dt)
                    h_close = b_avg
                    h_close_dt = ""
                    if sym in history_frames:
                        hdf = history_frames[sym]
                        post = hdf[hdf.index >= pd.to_datetime(w_date).tz_localize(None)]
                        if not post.empty and 'Close' in post.columns:
                            peak_idx = post['Close'].idxmax()
                            h_close = float(post['Close'].loc[peak_idx])
                            h_close_dt = pd.to_datetime(peak_idx).strftime('%d %b %y')
                    blk_rows.append({
                        'Block': b['block_name'],
                        'Wtd. Date': w_date.strftime('%d %b %Y'),
                        'Qty': b_qty,
                        'Avg Entry': b_avg,
                        'LTP': ltp,
                        'Peak Close': h_close,
                        'Peak Date': h_close_dt,
                        'Capital': b_inv,
                        'P&L': b_pnl,
                        'Return %': (b_pnl / b_inv * 100) if b_inv else 0,
                        'XIRR %': bxirr,
                    })
                st.dataframe(
                    pd.DataFrame(blk_rows).style.format({
                        'Avg Entry': '₹{:.2f}',
                        'LTP': '₹{:.2f}',
                        'Peak Close': '₹{:.2f}',
                        'Capital': '₹{:.2f}',
                        'P&L': '₹{:.2f}',
                        'Return %': '{:.2f}%',
                        'XIRR %': '{:.2f}%',
                    }), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 2: WATCHLIST SCANNER
# ══════════════════════════════════════════════════════════════════════════════
elif selected_nav == "🎯 Watchlist Scanner":
    _sl("TECHNICAL MATRIX")
    _st("Watchlist Opportunity Scanner")

    def _score_color(val):
        try:
            v = int(val)
            if v >= 5: return "color:#ff5252;font-weight:700"
            if v >= 4: return "color:#ff6d00;font-weight:700"
            if v == 3: return "color:#ffab00;font-weight:700"
            if v == 2: return "color:#ffe57f;font-weight:600"
        except Exception:
            pass
        return "color:#555d72"

    def _chg_color(val):
        try:
            v = float(str(val).replace("%", ""))
            if v > 0: return "color:#00e676;font-weight:600"
            if v < 0: return "color:#ff5252;font-weight:600"
        except Exception:
            pass
        return ""

    fmt_wl = {
        "LTP": "₹{:.2f}",
        "Day Chg %": "{:.2f}%",
        "RSI": "{:.1f}",
        "200 DMA": "₹{:.2f}",
        "BB Low": "₹{:.2f}",
        "52W High": "₹{:.2f}",
        "vs 52W %": "{:.2f}%",
        "ATH": "₹{:.2f}",
        "vs ATH %": "{:.2f}%",
    }
    fmt_held = {**fmt_wl, "Avg Price": "₹{:.2f}", "Last Block Avg": "₹{:.2f}", "Next Trigger": "₹{:.2f}"}

    def _render_wl_table(pool, err_msg, label):
        if err_msg and not pool:
            st.warning(f"Note ({label}): {err_msg}")
        if not pool:
            st.info(f"No tickers found for {label}.")
            return
        rows = []
        for sym in pool:
            if sym not in summary_feeds:
                continue
            m = summary_feeds[sym]
            score, reasons = compute_score(m, blocks=None)
            rows.append({
                "Symbol": sym,
                "Score": score,
                "Signal": score_label(score),
                "Reasons": reasons,
                "LTP": m["LTP"],
                "Day Chg %": m["Day_Change_Pct"],
                "RSI": m["RSI"],
                "200 DMA": m["200_DMA"],
                "BB Low": m["BB_LOW"],
                "52W High": m["52W_High"],
                "vs 52W %": m["vs_52W_Pct"],
                "ATH": m["ATH"],
                "vs ATH %": m["ATH_Dist_Pct"],
            })

        if not rows:
            st.info("No instruments ready to display.")
            return

        df_wl = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
        st.dataframe(
            df_wl.style
                 .format(fmt_wl)
                 .map(_score_color, subset=["Score"])
                 .map(_chg_color, subset=["Day Chg %"]),
            use_container_width=True, hide_index=True
        )

    wl_tabs = st.tabs(["01 Stocks", "02 ETFs", "03 Nifty 50", "04 Nifty Next 50"])
    with wl_tabs[0]:
        _render_wl_table(filtered_stock_pool, stk_err, "Stocks")
    with wl_tabs[1]:
        _render_wl_table(filtered_etf_pool, etf_err, "ETFs")
    with wl_tabs[2]:
        _render_wl_table(filtered_nifty50_pool, n50_err, "Nifty 50")
    with wl_tabs[3]:
        _render_wl_table(filtered_niftynext50_pool, nn50_err, "Nifty Next 50")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 3: TECHNICAL CHARTS
# ══════════════════════════════════════════════════════════════════════════════
elif selected_nav == "📈 Technical Charts":
    _sl("TECHNICAL MATRIX")
    _st("Interactive Chart Matrix")

    if not total_download_pool:
        st.info("No instruments tracked.")
    else:
        sel = st.selectbox("Select Instrument", sorted(total_download_pool), label_visibility="visible")

        if sel in history_frames and not history_frames[sel].empty:
            df_full = history_frames[sel]

            ranges = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "3Y": 365*3, "5Y": 365*5, "All": None}
            if "chart_range" not in st.session_state:
                st.session_state.chart_range = "1Y"
            
            r_cols = st.columns(len(ranges))
            for _ri, (_rl, _rd) in enumerate(ranges.items()):
                if r_cols[_ri].button(_rl, key="range_" + _rl, use_container_width=True):
                    st.session_state.chart_range = _rl
                    st.rerun()

            _sel_days = ranges[st.session_state.chart_range]
            if _sel_days is not None:
                _cutoff = pd.Timestamp.now() - pd.Timedelta(days=_sel_days)
                df = df_full[df_full.index >= _cutoff].copy()
                if df.empty: df = df_full.copy()
            else:
                df = df_full.copy()

            if PLOTLY_AVAILABLE:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.05, row_heights=[0.72, 0.28])

                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price',
                    line=dict(color='#00e5ff', width=1.8),
                    fill='tozeroy', fillcolor='rgba(0,229,255,0.04)'), row=1, col=1)
                
                if '200_DMA' in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df['200_DMA'], name='200 DMA',
                        line=dict(color='#ffab00', width=1.2, dash='dot')), row=1, col=1)
                if 'BB_LOW' in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOW'], name='BB Lower',
                        line=dict(color='#ff1744', width=1.1, dash='dash')), row=1, col=1)

                if sel in open_data and open_data[sel]['remaining_qty'] > 0:
                    blks = split_lots_into_blocks(open_data, BLOCK_SIZE).get(sel, [])
                    bx = []; by = []; bt = []
                    for b in blks:
                        bdf = pd.DataFrame(b['lots'])
                        b_avg = bdf['price'].sum() / b['qty']
                        bxirr = calculate_xirr([-lot['price'] for lot in b['lots']] + [b['qty'] * b_avg],
                                               [lot['date'] for lot in b['lots']] + [today_dt])
                        wd = get_xirr_weighted_buy_date(b['lots'], b['qty'] * b_avg, bxirr, today_dt)
                        bx.append(wd)
                        by.append(b_avg)
                        bt.append(f"{b['block_name']} ₹{b_avg:.2f}")
                    fig.add_trace(go.Scatter(x=bx, y=by, mode='markers+text',
                        marker=dict(symbol='triangle-up', size=12, color='#00e676',
                                    line=dict(width=1, color='#00c853')),
                        text=[b['block_name'] for b in blks],
                        textposition='top center',
                        textfont=dict(size=8, color='#00e676', family='JetBrains Mono'),
                        hovertext=bt, hoverinfo='text', name='Entries'), row=1, col=1)

                if 'RSI' in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                        line=dict(color='#00e5ff', width=1.3),
                        fill='tozeroy', fillcolor='rgba(0,229,255,0.04)'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[40, 40],
                        line=dict(color='#ff1744', width=.8, dash='dash'), showlegend=False), row=2, col=1)
                    fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[60, 60],
                        line=dict(color='#00e676', width=.8, dash='dash'), showlegend=False), row=2, col=1)

                fig.update_layout(
                    height=540, margin=dict(l=0, r=0, t=8, b=8),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='JetBrains Mono', size=9, color='#7b82a0'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                        font=dict(size=8), bgcolor='rgba(0,0,0,0)'),
                    hovermode="x unified",
                )
                fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.04)', zeroline=False)
                fig.update_yaxes(title_text="Price ₹", row=1, col=1)
                fig.update_yaxes(title_text="RSI", range=[10, 90], row=2, col=1)

                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.line_chart(df[['Close']])
                st.caption("Plotly is optional for high-precision candlestick view.")
        else:
            st.info("Insufficient historical candles for this instrument.")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 4: CLOSED TRADES & REALIZED P&L
# ══════════════════════════════════════════════════════════════════════════════
elif selected_nav == "📜 Closed Trades & P&L":
    _sl("TRANSACTION HISTORY")
    _st("Closed Trade Ledger")

    total_dep = 0.0; total_rel = 0.0; total_pnl = 0.0; ret_pct = 0.0; overall_closed_xirr = 0.0
    real_rows = []

    if not closed_trades:
        st.info("No closed positions recorded.")
    else:
        ct_df = pd.DataFrame(closed_trades)
        total_dep = ct_df['Buy Value'].sum()
        total_rel = ct_df['Sell Value'].sum()
        total_pnl = ct_df['Profit'].sum()
        ret_pct   = (total_pnl / total_dep * 100) if total_dep else 0.0

        _all_cl_cfs   = []
        _all_cl_dates = []
        for _, row in ct_df.iterrows():
            _all_cl_cfs.append(-row['Buy Value']); _all_cl_dates.append(row['Buy Date'])
            _all_cl_cfs.append(row['Sell Value']); _all_cl_dates.append(row['Sell Date'])
        overall_closed_xirr = calculate_xirr(_all_cl_cfs, _all_cl_dates)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Capital Deployed",  f"₹{total_dep:,.0f}")
        r2.metric("Capital Realized",  f"₹{total_rel:,.0f}")
        r3.metric("Booked P&L",        f"₹{total_pnl:,.0f}", f"{'+' if total_pnl >= 0 else ''}{ret_pct:.2f}%")
        r4.metric("Closed XIRR",       f"{overall_closed_xirr:.2f}%")

        _div()
        _sl("BY SYMBOL")
        _st("Realized Return Summary")

        for sym in ct_df['Symbol'].unique():
            sdf = ct_df[ct_df['Symbol'] == sym]
            ti  = sdf['Buy Value'].sum()
            tr  = sdf['Sell Value'].sum()
            tp  = sdf['Profit'].sum()

            sym_cfs   = []
            sym_dates = []
            for _, row in sdf.iterrows():
                sym_cfs.append(-row['Buy Value']); sym_dates.append(row['Buy Date'])
                sym_cfs.append(row['Sell Value']); sym_dates.append(row['Sell Date'])
            sym_xirr = calculate_xirr(sym_cfs, sym_dates)

            avg_hold = (
                pd.to_datetime(sdf['Sell Date'].astype(np.int64).mean()) -
                pd.to_datetime(sdf['Buy Date'].astype(np.int64).mean())
            ).days

            real_rows.append({
                'Symbol':      sym,
                'Trades':      len(sdf),
                'Qty':         sdf['Qty'].sum(),
                'Deployed ₹':  ti,
                'Realized ₹':  tr,
                'Net P&L ₹':   tp,
                'Return %':    (tp / ti * 100) if ti else 0,
                'Avg Hold':    avg_hold,
                'XIRR %':      sym_xirr,
            })

        def _pnl_color(val):
            try:
                v = float(str(val).replace('₹','').replace('%','').replace(',',''))
                if v > 0: return 'color:' + T['g'] + ';font-weight:600'
                if v < 0: return 'color:' + T['r'] + ';font-weight:600'
            except Exception: pass
            return ''

        _real_fmt = {
            'Deployed ₹': '₹{:.2f}', 'Realized ₹': '₹{:.2f}',
            'Net P&L ₹':  '₹{:.2f}', 'Return %':   '{:.2f}%',
            'XIRR %':     '{:.2f}%',
        }

        def _render_realized_table(df):
            st.dataframe(
                df.style
                  .format(_real_fmt)
                  .map(_pnl_color, subset=['Net P&L ₹', 'Return %', 'XIRR %']),
                use_container_width=True, hide_index=True
            )

        real_df  = pd.DataFrame(real_rows)
        df_gain  = real_df[real_df['Net P&L ₹'] >  0].sort_values('Net P&L ₹', ascending=False).reset_index(drop=True)
        df_loss  = real_df[real_df['Net P&L ₹'] <= 0].sort_values('Net P&L ₹', ascending=True).reset_index(drop=True)

        _sl("PROFITABLE SYMBOLS")
        if not df_gain.empty:
            _render_realized_table(df_gain)
        else:
            st.info("No symbols with positive realized P&L.")

        _div()
        _sl("LOSS-MAKING SYMBOLS")
        if not df_loss.empty:
            _render_realized_table(df_loss)
        else:
            st.info("No symbols with negative realized P&L.")

        _div()
        with st.expander("Full trade log"):
            st.dataframe(ct_df.style.format({
                'Buy Price': '₹{:.2f}', 'Sell Price': '₹{:.2f}',
                'Buy Value': '₹{:.2f}', 'Sell Value': '₹{:.2f}',
                'Profit': '₹{:.2f}', 'Profit %': '{:.2f}%'
            }), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 5: SWING RADAR
# ══════════════════════════════════════════════════════════════════════════════
elif selected_nav == "⚡ Swing Radar":
    _sl("SWING ANALYSIS")
    _st("Floor & Peak Pivot Radar")

    SWING_THRESHOLD = st.slider(
        "Swing threshold %", min_value=3, max_value=15, value=7, step=1,
        help="Minimum % move to confirm a new pivot and trigger a signal"
    )

    swing_pool = tuple(sorted(set(holding_symbols + filtered_stock_pool + filtered_etf_pool + filtered_nifty50_pool + filtered_niftynext50_pool)))
    swing_rows = []

    if not swing_pool:
        st.info("No symbols to analyse. Add holdings or watchlist entries.")
    else:
        with st.spinner("Computing swing pivots…"):
            swing_data = compute_swing_analysis_cached(swing_pool, SWING_THRESHOLD)

        if not swing_data:
            st.info("No swing data available.")
        else:
            for sd in swing_data:
                sym = sd["sym"]
                ltp = summary_feeds.get(sym, {}).get("LTP", sd["extreme_p"])
                w52 = summary_feeds.get(sym, {}).get("52W_High", sd["extreme_p"])
                ep = sd["extreme_p"]
                status = sd["status"]

                dist_pct = (ltp - ep) / ep * 100 if ep else 0.0
                scope_pct = (w52 - ltp) / ltp * 100 if ltp else 0.0
                pivot_diff_pct = 0.0
                if sd.get("pivots"):
                    try:
                        import re
                        p1_str = sd["pivots"][0]
                        match = re.search(r"₹([\d.]+)", p1_str)
                        if match:
                            p1_price = float(match.group(1))
                            pivot_diff_pct = (ltp - p1_price) / p1_price * 100 if p1_price else 0.0
                    except Exception:
                        pass

                abs_dist = abs(dist_pct)
                if abs_dist >= SWING_THRESHOLD:
                    if status == "FLOOR" and dist_pct > 0:
                        signal = "★ BUY ZONE"
                    elif status == "PEAK" and dist_pct < 0:
                        signal = "★ SELL ZONE"
                    else:
                        signal = "★ SIGNAL"
                elif abs_dist >= 3:
                    signal = "⏳ WATCH"
                else:
                    signal = "➔ TRENDING"

                swing_rows.append({
                    "Symbol": sym,
                    "Signal": signal,
                    "Leg": status,
                    "Extreme ₹": ep,
                    "Extreme Date": sd["extreme_date"].strftime("%d %b %y"),
                    "LTP": ltp,
                    "Dist Extreme %": dist_pct,
                    "Scope Left %": scope_pct,
                    "Diff Pivot1 %": pivot_diff_pct,
                    "Pivots": sd["pivot_count"],
                    "P1 (latest)": sd["pivots"][0] if len(sd["pivots"]) > 0 else "",
                    "P2": sd["pivots"][1] if len(sd["pivots"]) > 1 else "",
                })

            df_sw = pd.DataFrame(swing_rows)
            st.dataframe(df_sw, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 6: UPLOAD TRADES & MANUAL ENTRY
# ══════════════════════════════════════════════════════════════════════════════
elif selected_nav == "📥 Upload & Manual Trades":
    _sl("EMERGENCY ENTRY")
    _st("Manually Add a Trade (Pending Official Confirmation)")

    st.caption("Use this only when a trade happened but the official Zerodha export isn't available yet.")

    with st.form("manual_trade_form", clear_on_submit=True):
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            m_symbol = st.text_input("Symbol", placeholder="e.g. RELIANCE").upper().strip()
        with mc2:
            m_type = st.selectbox("Trade Type", ["buy", "sell"])
        with mc3:
            m_qty = st.number_input("Quantity", min_value=1, step=1, value=1)
        with mc4:
            m_price = st.number_input("Price", min_value=0.01, step=0.05, value=100.00, format="%.2f")

        m_date = st.date_input("Trade Date", value=datetime.now().date())
        manual_submit = st.form_submit_button("➕ Add Emergency Trade", type="primary")

        if manual_submit:
            if not m_symbol:
                st.error("Symbol is required.")
            else:
                _ts = datetime.now()
                _manual_trade_id = "MANUAL-" + _ts.strftime("%Y%m%d%H%M%S%f")
                _exec_time = datetime.combine(m_date, _ts.time())

                new_manual_row = {
                    "Sr": "",
                    "Symbol": m_symbol,
                    "ISIN": "",
                    "Trade Date": pd.Timestamp(m_date),
                    "Exchange": "NSE",
                    "Segment": "EQ",
                    "Series": "EQ",
                    "Trade Type": m_type,
                    "Auction": "false",
                    "Quantity": int(m_qty),
                    "Price": float(m_price),
                    "Trade ID": _manual_trade_id,
                    "Order ID": _manual_trade_id,
                    "Order Execution Time": _exec_time,
                }

                _existing = st.session_state.manual_trades_df
                if _existing.empty:
                    _updated_manual = pd.DataFrame([new_manual_row])
                else:
                    _updated_manual = pd.concat([_existing, pd.DataFrame([new_manual_row])], ignore_index=True)

                with st.spinner("Saving manual trade…"):
                    ok, err, medium = commit_file_to_github(
                        _updated_manual.to_csv(index=False),
                        MANUAL_CSV_FILENAME,
                        st.session_state.manual_trades_sha,
                        message=f"Manual trade: {m_type} {m_qty} {m_symbol} @ {m_price}"
                    )

                if ok:
                    st.success(f"Added manual {m_type} of {m_qty} {m_symbol} @ ₹{m_price:.2f} ({medium})")
                    fetch_manual_trades_cached.clear()
                    compute_indicators_cached.clear()
                    fetch_live_ltp_batch_cached.clear()
                    st.rerun()
                else:
                    st.error(f"Failed to save manual trade: {err}")

    if not st.session_state.manual_trades_df.empty:
        _div()
        _sl("PENDING MANUAL TRADES")
        _manual_display = st.session_state.manual_trades_df[["Symbol", "Trade Type", "Quantity", "Price", "Trade Date"]].copy()
        _manual_display["Trade Date"] = pd.to_datetime(_manual_display["Trade Date"]).dt.strftime("%d %b %Y")
        st.dataframe(_manual_display, use_container_width=True, hide_index=True)

    _div()
    _sl("TRADE UPLOAD")
    _st("Merge New Zerodha Export into Master Tradebook")

    uploaded_file = st.file_uploader(
        "Upload Zerodha tradebook export (tradebook-UFQ740-EQ.csv)",
        type=["csv"],
        help="Dedup by Trade Date + Trade ID. BSE converted to NSE automatically."
    )

    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)
            new_df.columns = [c.strip() for c in new_df.columns]
        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")
            st.stop()

        master_df = tradebook_df.copy()
        st.info(f"Loaded master with {len(master_df)} existing trades.")

        # ZERODHA COLUMN RENAMING
        ZERODHA_COL_MAP = {
            "symbol": "Symbol", "isin": "ISIN", "trade_date": "Trade Date",
            "exchange": "Exchange", "segment": "Segment", "series": "Series",
            "trade_type": "Trade Type", "auction": "Auction", "quantity": "Quantity",
            "price": "Price", "trade_id": "Trade ID", "order_id": "Order ID",
            "order_execution_time": "Order Execution Time",
        }
        new_df.rename(columns=ZERODHA_COL_MAP, inplace=True)
        required = ["Exchange", "Trade Date", "Trade ID", "Symbol", "Trade Type", "Quantity", "Price", "Order Execution Time"]
        missing = [c for c in required if c not in new_df.columns]
        if missing:
            st.error(f"Missing required columns in CSV: {missing}")
            st.stop()

        new_df["Exchange"] = new_df["Exchange"].astype(str).str.strip().str.upper().replace("BSE", "NSE")
        new_df["Trade Date"] = pd.to_datetime(new_df["Trade Date"], format="mixed").dt.strftime("%Y-%m-%d")
        new_df["Trade ID"] = new_df["Trade ID"].astype(str).str.strip()

        master_df["Trade Date"] = pd.to_datetime(master_df["Trade Date"], format="mixed").dt.strftime("%Y-%m-%d")
        master_df["Trade ID"] = master_df["Trade ID"].astype(str).str.strip()

        master_keys = set(zip(master_df["Trade Date"], master_df["Trade ID"]))
        new_df["_key"] = list(zip(new_df["Trade Date"], new_df["Trade ID"]))

        fresh_df = new_df[~new_df["_key"].isin(master_keys)].drop(columns=["_key"]).copy()
        dup_count = len(new_df) - len(fresh_df)

        st.metric("New trades detected", len(fresh_df), delta=f"{dup_count} duplicates skipped")

        if len(fresh_df) > 0:
            st.dataframe(fresh_df, use_container_width=True, hide_index=True)
            if st.button("Confirm & Merge into Master Tradebook", type="primary"):
                merged = pd.concat([master_df, fresh_df], ignore_index=True)
                with st.spinner("Writing master tradebook…"):
                    ok, err, medium = commit_file_to_github(
                        merged.to_csv(index=False), CSV_FILENAME,
                        message=f"Upload: merged {len(fresh_df)} trades"
                    )
                if ok:
                    st.success(f"Master tradebook successfully updated! ({medium})")
                    fetch_tradebook_cached.clear()
                    compute_indicators_cached.clear()
                    fetch_live_ltp_batch_cached.clear()
                    st.rerun()
                else:
                    st.error(f"Commit failed: {err}")
        else:
            st.info("Master tradebook is already up-to-date with this export.")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 7: DATA EXPORT
# ══════════════════════════════════════════════════════════════════════════════
elif selected_nav == "💾 Data Export":
    _sl("STATE EXPORT")
    _st("Export Complete Terminal State (Excel / GitHub)")

    st.caption("Generate a snapshot of your entire calculated terminal state (Holdings, Indicators, Closed Trades, Blocks, and Swing Radar) into multi-sheet `data.xlsx`.")

    output = io.BytesIO()
    try:
        # Build worksheets
        h_rows = []
        for sym, m in holding_meta.items():
            h_rows.append({
                "Symbol": sym,
                "Remaining Qty": open_data[sym]["remaining_qty"],
                "Avg Price": open_data[sym]["avg_price"],
                "Invested Amount": m["invested"],
                "LTP": m["ltp"],
                "Current Value": m["cur_val"],
                "Day Change %": m["day_chg"],
                "P&L (₹)": m["pnl_amt"],
                "P&L (%)": m["pnl_pct"],
                "Realized P&L": m["realized_pnl"],
                "Open XIRR": m["xirr_open"],
                "Combined XIRR": m["xirr_combined"]
            })
        df_holdings = pd.DataFrame(h_rows)

        blk_rows_export = []
        for sym, b_list in blocks_map.items():
            ltp = summary_feeds.get(sym, {}).get("LTP", open_data[sym]["avg_price"])
            for b in b_list:
                bdf = pd.DataFrame(b['lots'])
                b_qty = b['qty']
                b_avg = bdf['price'].sum() / b_qty
                b_inv = bdf['price'].sum()
                b_cur = b_qty * ltp
                b_pnl = b_cur - b_inv
                b_pct = b_pnl / b_inv * 100 if b_inv else 0
                bxirr = calculate_xirr([-lot['price'] for lot in b['lots']] + [b_cur],
                                       [lot['date'] for lot in b['lots']] + [today_dt])
                w_date = get_xirr_weighted_buy_date(b['lots'], b_cur, bxirr, today_dt)
                blk_rows_export.append({
                    "Symbol": sym, "Block Name": b["block_name"], "Wtd Date": w_date.strftime("%Y-%m-%d"),
                    "Qty": b_qty, "Avg Entry": b_avg, "Capital": b_inv, "LTP": ltp,
                    "P&L (₹)": b_pnl, "Return %": b_pct, "Block XIRR": bxirr
                })
        df_blocks = pd.DataFrame(blk_rows_export)
        df_closed = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame()
        df_indicators = pd.DataFrame.from_dict(summary_feeds, orient='index').reset_index().rename(columns={"index": "Symbol"})

        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_holdings.to_excel(writer, sheet_name='Holdings', index=False)
            if not df_blocks.empty:
                df_blocks.to_excel(writer, sheet_name='Blocks', index=False)
            if not df_closed.empty:
                df_closed.to_excel(writer, sheet_name='Closed Trades', index=False)
            df_indicators.to_excel(writer, sheet_name='Indicators', index=False)

        excel_bytes = output.getvalue()
        excel_ready = True
    except Exception as e:
        excel_ready = False
        st.error(f"Error compiling Excel workbook: {e}")

    if excel_ready:
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button(
                label="📥 Download data.xlsx to Browser",
                data=excel_bytes,
                file_name=f"lofo_terminal_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary"
            )
        with col_exp2:
            if st.button("☁ Push data.xlsx to GitHub", use_container_width=True):
                with st.spinner("Pushing to GitHub repository…"):
                    b64 = base64.b64encode(excel_bytes).decode('utf-8')
                    ok, err, medium = commit_file_to_github(
                        b64, "data.xlsx", message="Export Terminal State Snapshot"
                    )
                    if ok:
                        st.success("✅ data.xlsx successfully pushed!")
                    else:
                        st.error(f"GitHub Push failed: {err}")
