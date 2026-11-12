"""
utils/data_processing.py - Financial & Technical Engines for LOFO Terminal
Pure algorithmic and data processing logic separated from UI for testability.
"""

import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:
    try:
        from backports.zoneinfo import ZoneInfo
    except ImportError:
        ZoneInfo = None

try:
    from scipy import optimize
    SCIPY_AVAILABLE = True
except ImportError:
    optimize = None
    SCIPY_AVAILABLE = False


def calculate_xirr(cashflows, dates):
    """
    Calculate Extended Internal Rate of Return (XIRR) as a percentage.
    Uses scipy.optimize.brentq if available, with a robust bisection fallback,
    and finally CAGR approximation.
    """
    if not cashflows or not dates or len(cashflows) < 2 or len(cashflows) != len(dates):
        return 0.0
    
    # Must have both positive and negative cashflows
    has_pos = any(cf > 0 for cf in cashflows)
    has_neg = any(cf < 0 for cf in cashflows)
    if not (has_pos and has_neg):
        return 0.0

    paired = sorted(zip(dates, cashflows), key=lambda x: x[0])
    ds = [p[0] for p in paired]
    cfs = [float(p[1]) for p in paired]
    t0 = ds[0]

    # Convert date offsets to years
    years = [(d - t0).days / 365.25 for d in ds]
    if max(years) <= 0:
        return 0.0

    def npv(rate):
        if rate <= -1.0:
            return float('inf')
        return sum(cf / ((1.0 + rate) ** yr) for cf, yr in zip(cfs, years))

    # 1. Try Scipy Brentq if installed
    if SCIPY_AVAILABLE and optimize is not None:
        try:
            return float(optimize.brentq(npv, -0.9999, 10.0, maxiter=100) * 100.0)
        except Exception:
            pass

    # 2. Pure Python Bisection Method
    try:
        low = -0.9999
        high = 10.0
        f_low = npv(low)
        f_high = npv(high)
        if f_low * f_high <= 0:
            for _ in range(100):
                mid = (low + high) / 2.0
                f_mid = npv(mid)
                if abs(f_mid) < 1e-6 or (high - low) < 1e-6:
                    return float(mid * 100.0)
                if f_low * f_mid < 0:
                    high = mid
                    f_high = f_mid
                else:
                    low = mid
                    f_low = f_mid
            return float(((low + high) / 2.0) * 100.0)
    except Exception:
        pass

    # 3. Fallback: Simple CAGR approximation
    try:
        inv = abs(sum(cf for cf in cfs if cf < 0))
        ret = sum(cf for cf in cfs if cf > 0)
        yrs = (ds[-1] - ds[0]).days / 365.25
        if yrs > 0 and inv > 0:
            cagr = ((ret / inv) ** (1.0 / yrs) - 1.0) * 100.0
            return float(max(-99.0, min(cagr, 1000.0)))
    except Exception:
        pass

    return 0.0


def get_xirr_weighted_buy_date(lots_list, current_valuation, target_xirr_pct, target_date):
    """
    Computes an effective weighted buy date corresponding to a target XIRR percentage.
    """
    if not lots_list:
        return target_date

    total_outflow = sum(lot['price'] for lot in lots_list)
    if total_outflow <= 0 or current_valuation <= 0:
        return min(lot['date'] for lot in lots_list)

    r = target_xirr_pct / 100.0
    ratio = current_valuation / total_outflow
    if abs(r) < 0.0001 or (1.0 + r) <= 0 or ratio <= 0:
        avg_days = sum((target_date - lot['date']).days for lot in lots_list) / len(lots_list)
        return target_date - pd.to_timedelta(avg_days, unit='D')

    try:
        t = np.log(ratio) / np.log(1.0 + r)
        return target_date - pd.to_timedelta(t * 365.25, unit='D')
    except Exception:
        return min(lot['date'] for lot in lots_list)


def process_lofo(trades_df):
    """
    Process trades using LOFO (Lowest In, First Out).
    Matches sell transactions against the lowest buy price lots first.
    Returns: (closed_trades: list, open_data: dict)
    """
    closed_trades = []
    open_data = {}

    if trades_df.empty or 'Symbol' not in trades_df.columns:
        return closed_trades, open_data

    for symbol in trades_df['Symbol'].unique():
        stdf = trades_df[trades_df['Symbol'] == symbol].copy()
        exchange = stdf['Exchange'].iloc[0] if (len(stdf) > 0 and 'Exchange' in stdf.columns) else "NSE"

        # Buys in execution time order
        buys = stdf[stdf['Trade Type'] == 'buy'].copy()
        if not buys.empty and 'Order Execution Time' in buys.columns:
            buys = buys.sort_values('Order Execution Time').reset_index(drop=True)
        else:
            buys = buys.reset_index(drop=True)

        sells = stdf[stdf['Trade Type'] == 'sell'].copy()
        if not sells.empty:
            sells['Date_Key'] = sells['Trade Date'].dt.normalize()
            clubbed = []
            for d_key, gdf in sells.groupby('Date_Key'):
                tot_qty = gdf['Quantity'].sum()
                avg_p = (gdf['Quantity'] * gdf['Price']).sum() / tot_qty if tot_qty > 0 else 0
                max_time = gdf['Order Execution Time'].max() if 'Order Execution Time' in gdf.columns else d_key
                clubbed.append({
                    'Trade Date': d_key,
                    'Order Execution Time': max_time,
                    'Quantity': tot_qty,
                    'Price': avg_p
                })
            sells_df = pd.DataFrame(clubbed).sort_values('Order Execution Time').reset_index(drop=True)
        else:
            sells_df = pd.DataFrame(columns=['Trade Date', 'Order Execution Time', 'Quantity', 'Price'])

        timeline = []
        for _, b in buys.iterrows():
            timeline.append({
                'type': 'buy',
                'date': b['Trade Date'],
                'time': b.get('Order Execution Time', b['Trade Date']),
                'qty': int(b['Quantity']),
                'price': float(b['Price'])
            })
        for _, s in sells_df.iterrows():
            timeline.append({
                'type': 'sell',
                'date': s['Trade Date'],
                'time': s.get('Order Execution Time', s['Trade Date']),
                'qty': int(s['Quantity']),
                'price': float(s['Price'])
            })
        timeline = sorted(timeline, key=lambda x: x['time'])

        lots = []
        realized_profit = 0.0
        realized_cf = []
        realized_dates = []

        for trans in timeline:
            if trans['type'] == 'buy':
                lots.append({
                    'date': trans['date'],
                    'order_time': trans['time'],
                    'quantity': trans['qty'],
                    'price': trans['price'],
                    'remaining_qty': trans['qty']
                })
                realized_cf.append(-trans['qty'] * trans['price'])
                realized_dates.append(trans['date'])
            elif trans['type'] == 'sell':
                rem = trans['qty']
                sell_price = trans['price']
                sell_date = trans['date']
                while rem > 0:
                    active = [l for l in lots if l['remaining_qty'] > 0]
                    if not active:
                        break
                    # LOFO: Lowest buy price first
                    tgt = min(active, key=lambda x: x['price'])
                    orig = tgt
                    mq = min(orig['remaining_qty'], rem)
                    bv = mq * orig['price']
                    sv = mq * sell_price
                    pnl = sv - bv
                    closed_trades.append({
                        'Symbol': symbol,
                        'Buy Date': orig['date'],
                        'Sell Date': sell_date,
                        'Days': (sell_date - orig['date']).days,
                        'Qty': mq,
                        'Buy Price': round(orig['price'], 2),
                        'Sell Price': round(sell_price, 2),
                        'Buy Value': round(bv, 2),
                        'Sell Value': round(sv, 2),
                        'Profit': round(pnl, 2),
                        'Profit %': round(pnl / bv * 100, 2) if bv else 0.0
                    })
                    realized_profit += pnl
                    realized_cf.append(sv)
                    realized_dates.append(sell_date)
                    orig['remaining_qty'] -= mq
                    rem -= mq
                lots = [l for l in lots if l['remaining_qty'] > 0]

        rqty = sum(l['remaining_qty'] for l in lots)
        inv_amt = sum(l['remaining_qty'] * l['price'] for l in lots) if rqty > 0 else 0.0
        avg_price = inv_amt / rqty if rqty > 0 else 0.0

        open_data[symbol] = {
            'symbol': symbol,
            'exchange': exchange,
            'lots': lots if rqty > 0 else [],
            'remaining_qty': rqty,
            'invested_amount': inv_amt,
            'avg_price': avg_price,
            'realized_profit': realized_profit,
            'realized_cf': realized_cf,
            'realized_dates': realized_dates
        }

    return closed_trades, open_data


def split_lots_into_blocks(open_data, block_size=8000):
    """
    Splits open lots into capital allocation blocks of approximately block_size.
    LIQUIDCASE is aggregated into a single block.
    """
    blocks_out = {}
    for sym, d in open_data.items():
        if d['remaining_qty'] == 0:
            continue
        sym_blocks = []
        if sym == "LIQUIDCASE":
            all_lots = [{'price': l['price'], 'date': l['date']}
                        for l in d['lots'] for _ in range(l['remaining_qty'])]
            if all_lots:
                sym_blocks.append({'block_name': "Block 1", 'qty': len(all_lots), 'lots': all_lots})
            blocks_out[sym] = sym_blocks
            continue

        flat = [{'price': l['price'], 'date': l['date']}
                for l in sorted(d['lots'], key=lambda x: x['order_time'])
                for _ in range(l['remaining_qty'])]
        b_idx = 1
        cq = 0
        cv = 0.0
        cl = []
        for share in flat:
            if cv + share['price'] > block_size:
                if cq > 0:
                    sym_blocks.append({'block_name': f"Block {b_idx}", 'qty': cq, 'lots': cl})
                    b_idx += 1
                cq = 0
                cv = 0.0
                cl = []
            cq += 1
            cv += share['price']
            cl.append(share)

        if cq > 0:
            lbl = f"Block {b_idx} (P)" if cv < block_size else f"Block {b_idx}"
            sym_blocks.append({'block_name': lbl, 'qty': cq, 'lots': cl})
        blocks_out[sym] = sym_blocks
    return blocks_out


def calculate_wilder_rsi(close_prices, period=14):
    """
    Computes Wilder's Exponential Relative Strength Index (RSI).
    """
    if len(close_prices) == 0:
        return pd.Series(dtype=float)
    delta = close_prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    ag = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    al = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    # Avoid division by zero
    rs = ag / al.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(100.0 if (gain.sum() > 0 and loss.sum() == 0) else 50.0)


def parse_tradebook(raw_df):
    """
    Cleans and standardizes raw tradebook export dataframe.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    raw = raw_df.copy()
    raw.columns = [str(c).strip() for c in raw.columns]

    for col in ["Symbol", "Exchange"]:
        if col in raw.columns:
            raw[col] = raw[col].astype(str).str.strip().str.upper()
    if "Trade Type" in raw.columns:
        raw["Trade Type"] = raw["Trade Type"].astype(str).str.strip().str.lower()
    if "Quantity" in raw.columns:
        raw["Quantity"] = pd.to_numeric(raw["Quantity"], errors="coerce").fillna(0).astype(int)
    if "Price" in raw.columns:
        raw["Price"] = pd.to_numeric(raw["Price"], errors="coerce").fillna(0.0)
    for col in ["Trade Date", "Order Execution Time"]:
        if col in raw.columns:
            raw[col] = pd.to_datetime(raw[col], format="mixed", errors="coerce")

    if "Quantity" in raw.columns and "Trade Type" in raw.columns:
        filtered = raw[(raw["Quantity"] > 0) & raw["Trade Type"].isin(["buy", "sell"])].copy()
        return filtered.reset_index(drop=True)
    return pd.DataFrame()


def compute_score(m, blocks=None):
    """
    Calculates technical signal score and reason tags for a market instrument.
    """
    score = 0
    reasons = []
    ltp = m.get("LTP", 0.0)

    if m.get("RSI", 50.0) < 40:
        score += 1
        reasons.append("RSI<40")
    if ltp < m.get("200_DMA", ltp):
        score += 1
        reasons.append("<200DMA")
    if ltp < m.get("BB_LOW", ltp):
        score += 1
        reasons.append("<BBLow")

    vs52 = m.get("vs_52W_Pct", m.get("ATH_Dist_Pct", 0.0))
    if vs52 <= -20:
        score += 2
        reasons.append("52W↓20%+")
    elif vs52 <= -10:
        score += 1
        reasons.append("52W↓10%+")

    if m.get("ATH_Dist_Pct", 0.0) <= -20:
        score += 1
        reasons.append("ATH↓20%+")

    if blocks is not None and len(blocks) > 0:
        def blk_date(b):
            return min(lot["date"] for lot in b["lots"])
        sblks = sorted(blocks, key=blk_date)

        def ref_block_avg(sorted_blocks):
            complete = [b for b in sorted_blocks if "(P)" not in b["block_name"]]
            ref = complete[-1] if complete else sorted_blocks[-1]
            lots = ref["lots"]
            return sum(l["price"] for l in lots) / len(lots), ref["block_name"]

        n = len(sblks)
        if n == 1:
            ref_avg, ref_name = ref_block_avg(sblks)
            if ltp <= ref_avg * 0.95:
                score += 1
                reasons.append(f"2nd▼5%@{ref_name}")
        else:
            ref_avg, ref_name = ref_block_avg(sblks)
            if ltp <= ref_avg * 0.97:
                score += 1
                reasons.append(f"Nth▼3%@{ref_name}")

    return score, " · ".join(reasons) if reasons else "—"


def score_label(score, has_blocks=False):
    """
    Returns visual badge text for a score.
    """
    if score >= 5:
        return "🔴 Strong"
    if score >= 3:
        return "🟠 Moderate"
    if score >= 2:
        return "🟡 Weak"
    return "⚪ Watch"


def detect_swing_pivots(df, threshold_pct=7.0):
    """
    Analyzes historical price series to detect floor and peak pivot reversals.
    df must contain 'Close' index by Datetime.
    """
    if df is None or df.empty or len(df) < 5 or 'Close' not in df.columns:
        return None

    prices = df["Close"]
    dates = df.index
    thr = threshold_pct / 100.0

    pivot_history = []
    last_pivot_price = float(prices.iloc[0])
    trend = 0
    curr_high = curr_low = last_pivot_price
    curr_high_date = dates[0]
    curr_low_date = dates[0]

    for i in range(1, len(prices)):
        price = float(prices.iloc[i])
        date = dates[i]

        if trend == 0:
            chg = (price - last_pivot_price) / last_pivot_price if last_pivot_price else 0
            if chg <= -thr:
                trend = -1
            elif chg >= thr:
                trend = 1

        if trend == -1:
            if price < curr_low:
                curr_low = price
                curr_low_date = date
            elif curr_low > 0 and (price - curr_low) / curr_low >= thr:
                pivot_history.append({"date": curr_low_date, "price": curr_low, "type": "FLOOR"})
                last_pivot_price = curr_low
                trend = 1
                curr_high = price
                curr_high_date = date

        elif trend == 1:
            if price > curr_high:
                curr_high = price
                curr_high_date = date
            elif curr_high > 0 and (curr_high - price) / curr_high >= thr:
                pivot_history.append({"date": curr_high_date, "price": curr_high, "type": "PEAK"})
                last_pivot_price = curr_high
                trend = -1
                curr_low = price
                curr_low_date = date

    if trend == 1:
        extreme_p = curr_high
        extreme_date = curr_high_date
        status_label = "PEAK"
    else:
        extreme_p = curr_low
        extreme_date = curr_low_date
        status_label = "FLOOR"

    recent = pivot_history[-5:][::-1]
    pivot_strs = []
    for idx, pv in enumerate(recent):
        prev_price = recent[idx + 1]["price"] if idx + 1 < len(recent) else pv["price"]
        move_pct = (pv["price"] - prev_price) / prev_price * 100.0 if prev_price else 0.0
        pivot_strs.append(
            f"{pv['type'][0]}  {pv['date'].strftime('%d%b%y')}  "
            f"₹{pv['price']:.2f}  ({move_pct:+.1f}%)"
        )

    return {
        "status": status_label,
        "extreme_p": float(extreme_p),
        "extreme_date": extreme_date,
        "pivots": pivot_strs,
        "pivot_count": len(pivot_history)
    }


def get_ist_now():
    """
    Returns current datetime in Indian Standard Time (Asia/Kolkata).
    """
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo("Asia/Kolkata"))
        except Exception:
            pass
    try:
        import pytz
        return datetime.now(pytz.timezone("Asia/Kolkata"))
    except Exception:
        return datetime.utcnow() + timedelta(hours=5, minutes=30)


def is_market_open(now_ist=None):
    """
    Returns True if Indian stock market (NSE/BSE) is open:
    Monday-Friday, 09:15 AM to 03:30 PM IST.
    """
    if now_ist is None:
        now_ist = get_ist_now()
    if now_ist.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    open_mins = 9 * 60 + 15
    close_mins = 15 * 60 + 30
    cur_mins = now_ist.hour * 60 + now_ist.minute
    return open_mins <= cur_mins <= close_mins


def apply_live_prices(feeds, live):
    """
    Updates indicator summary feeds dictionary with live prices and recomputes distance metrics.
    """
    for sym, lv in live.items():
        if sym not in feeds:
            continue
        ltp = lv.get("ltp", feeds[sym].get("LTP", 0.0))
        ath = feeds[sym].get("ATH", ltp)
        w52 = feeds[sym].get("52W_High", ath)
        feeds[sym]["LTP"] = ltp
        feeds[sym]["Day_Change_Pct"] = lv.get("day_chg_pct", 0.0)
        feeds[sym]["ATH_Dist_Pct"] = ((ltp - ath) / ath * 100.0) if ath else 0.0
        feeds[sym]["vs_52W_Pct"] = ((ltp - w52) / w52 * 100.0) if w52 else 0.0

