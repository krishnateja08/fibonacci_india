"""
Fibonacci Swing Trading Plan Generator
=======================================
Fetches OHLC price history for a list of tickers, detects the most recent
significant swing leg (swing low -> swing high, or swing high -> swing low),
computes the standard Fibonacci retracement levels (23.6 / 38.2 / 50.0 /
61.8 / 78.6 %) plus the 161.8% extension target, builds an Entry / Stop-Loss
/ Take-Profit plan, and renders everything into a single clean HTML report.

INSTALL (one time):
    pip install yfinance pandas numpy

RUN:
    python fib_swing_trade_plan.py

OUTPUT:
    fibonacci_trade_plan.html  (open it in any browser)

You can freely edit the TICKERS list, LOOKBACK_DAYS, and SWING_ORDER below.
"""

import os
import sys
import json
import datetime as dt
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import pandas as pd
import yfinance as yf

# Directory the script itself lives in — output is always written here,
# regardless of what folder you happen to run the command from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_HTML = os.path.join(SCRIPT_DIR, "fibonacci_trade_plan.html")

# ──────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────

# How many calendar days of daily price history to pull per stock.
LOOKBACK_DAYS = 220

# Zigzag reversal threshold: a new swing pivot is only confirmed once price
# reverses by at least this % from the running high/low. This is what makes
# the swing detection follow the *real* structural swing (the one visible
# on a chart) instead of getting stuck on small day-to-day noise wiggles.
ZIGZAG_PCT = 5.0

# Kept only as a fallback knob for extremely short/quiet price histories
# where the zigzag can't confirm even one full swing.
SWING_ORDER = 8

# Which confirmed swing leg to trade:
#   "recent"  -> the most recently completed leg (last two confirmed
#                pivots). Freshest structure, best for near-term entries,
#                but a leg can be small if price has been choppy lately.
#   "largest" -> the single biggest leg (by price range) found anywhere
#                in the whole LOOKBACK_DAYS window. Captures the major
#                structural swing (e.g. a big top-to-bottom move) even if
#                it happened weeks ago and smaller legs have formed since.
SWING_MODE = "recent"

# Fibonacci ratios used throughout.
FIB_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
FIB_EXTENSION = 1.618

# Long-term trend filter: a swing-leg "Uptrend"/"Downtrend" label only tells
# you the direction of the *last leg* — it says nothing about the macro
# trend. A stock in a multi-month downtrend can throw a 5%+ bounce that
# zigzag happily calls an "Uptrend" leg, right into heavy overhead
# resistance. We cross-check the leg direction against a long SMA and flag
# (not silently discard) any leg that's fighting the macro trend.
SMA_PERIOD = 200
# Extra calendar days fetched (on top of LOOKBACK_DAYS) purely so the SMA
# has enough trading history to compute. 200 trading days needs roughly
# ~290 calendar days; we pad further for holidays/weekends.
SMA_FETCH_BUFFER_DAYS = 380

# Volume confirmation: a pullback on volume BELOW its own recent average
# suggests a lack-of-demand drift rather than aggressive institutional
# distribution tearing through the zone. High-volume pullbacks are flagged
# as higher risk of slicing straight through to the stop.
VOLUME_SMA_PERIOD = 20

# Broad market regime filter: individual-stock Fibonacci pullbacks fail
# far more often when the wider market is itself in a correction, even if
# the stock's own chart looks clean. We check one index once per run and
# tag every plan with whether its leg direction agrees with the tape.
MARKET_INDEX_TICKER = "^NSEI"   # Nifty 50
MARKET_SMA_PERIOD = 50

# Momentum context: 14-period Wilder RSI on daily closes. Not used to
# gate the trade status/zone logic — shown purely as an extra read so a
# Fib zone touch that's happening on a deeply oversold/overbought RSI
# can be weighed differently than the same zone touch with neutral RSI.
RSI_PERIOD = 14

TICKERS: List[str] = [
    # ── Rank 1-10 by Volume ──
    "ADANIPOWER.NS", "INFY.NS", "WIPRO.NS", "ETERNAL.NS",
    "JIOFIN.NS", "HDFCBANK.NS", "UNIONBANK.NS", "TATASTEEL.NS",
    "KOTAKBANK.NS", "VEDL.NS",
    # ── Rank 11-20 ──
    "CANBK.NS", "ITC.NS", "COALINDIA.NS", "IRFC.NS",
    "ICICIBANK.NS", "SBIN.NS", "HINDZINC.NS", "VBL.NS",
    "ADANIGREEN.NS", "ONGC.NS",
    # ── Rank 21-30 ──
    "RELIANCE.NS", "BEL.NS", "PNB.NS", "MOTHERSON.NS",
    "HCLTECH.NS", "BPCL.NS", "POWERGRID.NS", "SUNPHARMA.NS",
    "GAIL.NS", "SHRIRAMFIN.NS",
    # ── Rank 31-40 ──
    "IOC.NS", "PFC.NS", "ADANIENSOL.NS", "BANKBARODA.NS",
    "TATAPOWER.NS", "BHARTIARTL.NS", "NTPC.NS", "TATACAP.NS",
    "TMPV.NS", "DRREDDY.NS",
    # ── Rank 41-50 ──
    "SBILIFE.NS", "TCS.NS", "RECLTD.NS", "HINDALCO.NS",
    "TMCV.NS", "CIPLA.NS", "CGPOWER.NS", "BAJFINANCE.NS",
    "GODREJCP.NS", "AMBUJACEM.NS",
    # ── Rank 51-60 ──
    "TECHM.NS", "AXISBANK.NS", "NESTLEIND.NS", "HDFCLIFE.NS",
    "MAXHEALTH.NS", "M&M.NS", "ADANIPORTS.NS", "MAZDOCK.NS",
    "ADANIENT.NS", "INDHOTEL.NS",
    # ── Rank 61-70 ──
    "LT.NS", "DLF.NS", "JSWSTEEL.NS", "HINDUNILVR.NS",
    "TRENT.NS", "LODHA.NS", "TATACONSUM.NS", "CHOLAFIN.NS",
    "JINDALSTEL.NS", "GRASIM.NS",
    # ── Rank 71-80 ──
    "HYUNDAI.NS", "HDFCAMC.NS", "UNITDSPR.NS", "TITAN.NS",
    "LTM.NS", "BAJAJFINSV.NS", "HAL.NS", "TVSMOTOR.NS",
    "INDIGO.NS", "ZYDUSLIFE.NS",
    # ── Rank 81-90 ──
    "MUTHOOTFIN.NS", "ENRIN.NS", "PIDILITIND.NS", "CUMMINSIND.NS",
    "BRITANNIA.NS", "MARUTI.NS", "ASIANPAINT.NS", "EICHERMOT.NS",
    "APOLLOHOSP.NS", "ULTRACEMCO.NS",
    # ── Rank 91-100 ──
    "ABB.NS", "DIVISLAB.NS", "SIEMENS.NS", "SOLARINDS.NS",
    "TORNTPHARM.NS", "DMART.NS", "BAJAJ-AUTO.NS", "BAJAJHLDNG.NS",
    "BOSCHLTD.NS", "SHREECEM.NS",
    # ── Sector ETFs (BEES) ──
    "NIFTYBEES.NS", "BANKBEES.NS", "ITBEES.NS", "AUTOBEES.NS",
    "PHARMABEES.NS", "GOLDBEES.NS", "SILVERBEES.NS",
]

# Major indices, analyzed the same way as individual stocks but kept in a
# separate list/tab — index Fibonacci levels are a macro/context read
# (broad market or sector structure), not something you'd place an entry
# order against the way you would an individual stock.
INDEX_TICKERS: List[str] = [
    "^NSEI",        # Nifty 50
    "^NSEBANK",     # Nifty Bank
    "^BSESN",       # BSE Sensex
    "^CNXIT",       # Nifty IT
    "^CNXPHARMA",   # Nifty Pharma
    "^CNXAUTO",     # Nifty Auto
    "^CNXFMCG",     # Nifty FMCG
    "^CNXMETAL",    # Nifty Metal
    "^CNXREALTY",   # Nifty Realty
    "^CNXENERGY",   # Nifty Energy
    "^CNXPSUBANK",  # Nifty PSU Bank
    "^CNXMEDIA",    # Nifty Media
]


# ──────────────────────────────────────────────────────────────────────────
# DATA MODEL
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class TradePlan:
    ticker: str
    ok: bool = True
    error: Optional[str] = None

    current_price: float = 0.0
    price_is_live: bool = False  # True = current_price came from a live/intraday quote, not the daily candle close
    chg_pct: float = 0.0
    spark: List[float] = field(default_factory=list)  # last ~20 closes for a mini sparkline
    swing_high: float = 0.0
    swing_low: float = 0.0
    swing_high_date: Optional[str] = None
    swing_low_date: Optional[str] = None
    trend: str = ""  # "Uptrend" or "Downtrend"
    extended_leg: bool = False  # True if leg end is a still-forming extreme, not a confirmed pivot
    rsi: Optional[float] = None  # 14-period Wilder RSI on daily closes; None if not enough history

    sma200: Optional[float] = None       # long-term trend filter (None if not enough history)
    counter_trend: bool = False          # True if the leg direction fights the 200-SMA trend

    vol_confirmed: Optional[bool] = None   # True = today's volume is below its 20d avg (healthy pullback)
    reversal_confirmed: bool = False       # True = today's candle shows a rejection in the trade direction
    market_aligned: Optional[bool] = None  # True = index regime agrees with this leg's direction
    confirmations_note: str = ""           # short human-readable summary shown in the report

    levels: dict = field(default_factory=dict)   # ratio -> price
    extension_target: float = 0.0

    entry_low: float = 0.0
    entry_high: float = 0.0
    stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    distance_to_entry: float = 0.0  # abs distance from current price to entry-zone midpoint

    status: str = ""
    status_class: str = ""  # css class for coloring
    status_label: str = ""  # short label used in the pulse strip / filter chips
    fib_zone: str = ""      # which fib retracement band current price sits in, e.g. "38.2% - 50.0%"


# ──────────────────────────────────────────────────────────────────────────
# CORE LOGIC
# ──────────────────────────────────────────────────────────────────────────

def fetch_history(ticker: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError("No price data returned (bad ticker or no internet).")
    # yfinance sometimes returns MultiIndex columns for single tickers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    return df


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> Optional[float]:
    """
    Standard 14-period Wilder RSI on daily closes. Uses Wilder's own
    smoothing (an EWM with alpha = 1/period) rather than a plain rolling
    average, since that's the convention almost every charting platform
    (TradingView included) uses — a plain-average RSI would read
    noticeably differently from what's on the chart.

    Returns None if there isn't enough history to seed the average
    (needs at least `period` price changes), rather than a misleading
    early-series value.
    """
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    last_gain = avg_gain.iloc[-1]
    last_loss = avg_loss.iloc[-1]
    if pd.isna(last_gain) or pd.isna(last_loss):
        return None
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    return float(100 - (100 / (1 + rs)))


def fetch_live_price(ticker: str, fallback: float):
    """
    Gets the most current live/intraday price for `ticker` so status and
    Fib-zone classification reflect where price actually is *right now*
    rather than the daily candle's Close — which, while the market is
    open, is really just "price as of the last completed daily bar" and
    can be one full price move stale versus a live chart (this is why a
    stock can look like it's sitting in one Fib zone on a live chart but
    another in a report generated from the daily Close).

    Tries yfinance's fast_info (a live last-traded-price quote) first,
    then falls back to a 1-minute intraday bar, then finally falls back
    to the caller-supplied daily Close if neither live source works
    (e.g. market closed for a while, ticker unsupported, network hiccup)
    — so callers always get *a* usable price, just flagged as delayed.

    Returns (price, is_live).
    """
    try:
        fi = yf.Ticker(ticker).fast_info
        last = None
        for key in ("last_price", "lastPrice"):
            try:
                last = fi[key]
            except Exception:
                last = getattr(fi, key, None)
            if last:
                break
        if last and last > 0:
            return float(last), True
    except Exception:
        pass

    try:
        intraday = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
        if isinstance(intraday.columns, pd.MultiIndex):
            intraday.columns = intraday.columns.get_level_values(0)
        intraday = intraday.dropna(subset=["Close"])
        if not intraday.empty:
            return float(intraday["Close"].iloc[-1]), True
    except Exception:
        pass

    return fallback, False


def zigzag_pivots(df: pd.DataFrame, pct: float):
    """
    Wick-based zigzag: walks the daily High/Low range (NOT the close) and
    only confirms a new pivot once price has reversed by at least `pct`%
    from the running extreme since the last confirmed pivot. Fibonacci
    levels are conventionally anchored to the actual price extremes a
    stock traded at intraday, so peaks are tracked off `High` and troughs
    off `Low` — using `Close` would understate a wick-driven high/low and
    shift every downstream level.

    Returns a list of (position, price, kind) tuples in chronological
    order, kind being 'H' or 'L'.
    """
    highs = df["High"].values
    lows = df["Low"].values
    n = len(highs)
    if n < 2:
        return []

    pivots = []
    trend = None  # 'up' while tracking toward a high, 'down' while tracking toward a low
    ext_pos, ext_val = 0, highs[0]
    # During the undetermined phase (before a direction is confirmed) we
    # must track the true running max/min independently — NOT a single
    # pointer that chases the latest price, since a moving target can never
    # be "5% away from itself" and the threshold would never trigger.
    run_max_pos, run_max_val = 0, highs[0]
    run_min_pos, run_min_val = 0, lows[0]

    for i in range(1, n):
        hi, lo = highs[i], lows[i]
        if trend is None:
            if hi > run_max_val:
                run_max_pos, run_max_val = i, hi
            if lo < run_min_val:
                run_min_pos, run_min_val = i, lo

            if lo <= run_max_val * (1 - pct / 100):
                trend = "down"
                pivots.append((run_max_pos, run_max_val, "H"))
                ext_pos, ext_val = i, lo
            elif hi >= run_min_val * (1 + pct / 100):
                trend = "up"
                pivots.append((run_min_pos, run_min_val, "L"))
                ext_pos, ext_val = i, hi
        elif trend == "up":
            if hi > ext_val:
                ext_pos, ext_val = i, hi
            elif lo <= ext_val * (1 - pct / 100):
                pivots.append((ext_pos, ext_val, "H"))
                trend = "down"
                ext_pos, ext_val = i, lo
        else:  # trend == "down"
            if lo < ext_val:
                ext_pos, ext_val = i, lo
            elif hi >= ext_val * (1 + pct / 100):
                pivots.append((ext_pos, ext_val, "L"))
                trend = "up"
                ext_pos, ext_val = i, hi

    # NOTE: we deliberately do NOT append the still-forming trailing extreme
    # as a pivot. Only fully confirmed reversals belong in this list — the
    # in-progress extreme is exactly what "current price" represents, and
    # conflating the two would collapse the swing high/low onto today's
    # price. The caller compares current price against the last two
    # *confirmed* pivots to see where it sits inside that structure.
    return pivots


def detect_last_swing(df: pd.DataFrame):
    """
    Find the swing leg to trade using a percentage-based zigzag over the
    full lookback window, so the result reflects a real structural swing
    (matching what you'd see by eye on a chart) rather than a small
    recent noise wiggle. Which leg is picked depends on SWING_MODE:
    "recent" (freshest completed leg) or "largest" (biggest-range leg
    anywhere in the window). Returns (swing_high, swing_high_date,
    swing_low, swing_low_date, trend, is_extended).

    `is_extended` is True when the leg was rolled forward onto a
    still-forming (unconfirmed) extreme rather than a fully confirmed
    zigzag reversal — see the roll-forward comment below. Callers should
    surface this rather than silently treating the leg as settled,
    since these levels can still move as price continues.
    """
    pivots = zigzag_pivots(df, ZIGZAG_PCT)
    highs_all = df["High"].values
    lows_all = df["Low"].values
    is_extended = False

    if len(pivots) >= 2:
        if SWING_MODE == "largest":
            # Scan every consecutive confirmed-pivot pair (each one is a
            # real monotonic leg) and keep the one with the biggest price
            # range, regardless of how long ago it happened.
            best_i, best_range = 0, -1.0
            for i in range(len(pivots) - 1):
                rng = abs(pivots[i + 1][1] - pivots[i][1])
                if rng > best_range:
                    best_range = rng
                    best_i = i
            pos_a, val_a, kind_a = pivots[best_i]
            pos_b, val_b, kind_b = pivots[best_i + 1]
        else:  # "recent"
            pos_a, val_a, kind_a = pivots[-2]
            pos_b, val_b, kind_b = pivots[-1]

        # `pivots` only contains CONFIRMED reversals. If price has kept
        # moving past the last confirmed pivot without yet reversing by
        # ZIGZAG_PCT%, that still-forming move is invisible to `pivots`,
        # and swing_high/swing_low would be stale — e.g. reporting an old
        # high from weeks ago while price has since rallied 20% past it
        # with no pullback. Roll the leg forward to the current running
        # extreme, but ONLY when that extreme actually breaks past the
        # OTHER pivot's value (val_a) — i.e. price has made a genuinely
        # new high/low beyond the whole established leg, not just a
        # normal bounce that's still inside it. A running extreme that
        # stays within the leg's existing range is exactly the kind of
        # in-range noise the zigzag threshold is designed to filter out
        # (this is what caused the May-5th high to get wrongly discarded
        # in favor of a smaller June bounce). Also only applies when the
        # chosen leg's end IS the most recent confirmed pivot — rolling
        # an older, already-superseded leg forward would just re-litigate
        # price action that's already captured by later pivots.
        is_latest_leg = (pos_b == pivots[-1][0])
        if is_latest_leg:
            # Use High for a new-high search, Low for a new-low search —
            # matching the wick-based pivots above instead of closes.
            if kind_b == "L":
                tail = highs_all[pos_b:]
                run_rel = int(np.argmax(tail))
            else:  # kind_b == "H"
                tail = lows_all[pos_b:]
                run_rel = int(np.argmin(tail))
            run_pos = pos_b + run_rel
            run_val = float(tail[run_rel])

            breaks_out = (run_val > val_a) if kind_b == "L" else (run_val < val_a)
            if run_pos > pos_b and breaks_out:
                pos_a, val_a, kind_a = pos_b, val_b, kind_b
                pos_b, val_b, kind_b = run_pos, run_val, ("H" if kind_b == "L" else "L")
                # The leg's new endpoint is an unconfirmed running extreme,
                # not a fully reversed zigzag pivot — flag it so the UI can
                # tell the user these levels may still shift.
                is_extended = True
    else:
        # Fallback for very short/quiet histories where the zigzag can't
        # confirm even one full swing at this threshold.
        n = len(highs_all)
        high_idx, low_idx = [], []
        # Use an adaptive window that shrinks near the start/end of the
        # series instead of a fixed SWING_ORDER buffer. A fixed buffer
        # (range(SWING_ORDER, n - SWING_ORDER)) silently excludes the most
        # recent `SWING_ORDER` bars from ever being chosen as a pivot,
        # which means a genuine swing high/low sitting in the last ~8
        # trading days would be structurally invisible to this fallback.
        for i in range(1, n - 1):
            radius = min(SWING_ORDER, i, n - 1 - i)
            hi_window = highs_all[i - radius:i + radius + 1]
            lo_window = lows_all[i - radius:i + radius + 1]
            if highs_all[i] == hi_window.max():
                high_idx.append(i)
            if lows_all[i] == lo_window.min():
                low_idx.append(i)
        if high_idx and low_idx:
            pos_a, val_a, kind_a = high_idx[-1], float(highs_all[high_idx[-1]]), "H"
            pos_b, val_b, kind_b = low_idx[-1], float(lows_all[low_idx[-1]]), "L"
        else:
            hi_pos, lo_pos = int(highs_all.argmax()), int(lows_all.argmin())
            pos_a, val_a, kind_a = hi_pos, float(highs_all[hi_pos]), "H"
            pos_b, val_b, kind_b = lo_pos, float(lows_all[lo_pos]), "L"

    if kind_a == "H":
        hi_pos, swing_high = pos_a, val_a
        lo_pos, swing_low = pos_b, val_b
    else:
        hi_pos, swing_high = pos_b, val_b
        lo_pos, swing_low = pos_a, val_a

    swing_high_date = str(df.index[hi_pos].date())
    swing_low_date = str(df.index[lo_pos].date())

    # Trend = direction of the leg that finished most recently.
    trend = "Uptrend" if lo_pos < hi_pos else "Downtrend"

    return swing_high, swing_high_date, swing_low, swing_low_date, trend, is_extended


def get_market_regime(index_ticker: str = MARKET_INDEX_TICKER,
                       sma_period: int = MARKET_SMA_PERIOD) -> Optional[dict]:
    """
    Fetches the broad market index once and checks whether it's trading
    above its own SMA. Returns {"price", "sma", "bullish"} or None if the
    index can't be fetched (non-fatal — callers should treat it as an
    optional extra check, not a hard requirement).
    """
    try:
        df = fetch_history(index_ticker, days=LOOKBACK_DAYS + SMA_FETCH_BUFFER_DAYS)
        if len(df) < sma_period:
            return None
        sma = df["Close"].rolling(sma_period).mean().iloc[-1]
        if pd.isna(sma):
            return None
        price = float(df["Close"].iloc[-1])
        return {"price": price, "sma": float(sma), "bullish": price > float(sma)}
    except Exception:
        return None


def compute_fib_zone(current_price: float, levels: dict, trend: str) -> str:
    """
    Returns which Fibonacci retracement band `current_price` currently sits
    in, e.g. "38.2% - 50.0%", based on the already-computed ratio->price
    `levels` dict. This is a pure "where is price right now" read of the
    Fib structure — independent of the Buy/Short/Waiting status logic —
    so it always reflects the exact zone regardless of trend/status.

    In an Uptrend, levels[0.0] is the swing high and price falls as the
    ratio increases toward levels[1.0] (the swing low). In a Downtrend
    it's the reverse: levels[0.0] is the swing low and price rises as the
    ratio increases toward levels[1.0] (the swing high).
    """
    if not levels:
        return "—"

    ratios = sorted(levels.keys())
    prices = [levels[r] for r in ratios]

    if trend == "Uptrend":
        # prices descend as ratio increases: prices[0] = high ... prices[-1] = low
        if current_price >= prices[0]:
            return "Above 0% (new high)"
        if current_price <= prices[-1]:
            return "Beyond 100% (fully retraced)"
        for i in range(len(ratios) - 1):
            hi_p, lo_p = prices[i], prices[i + 1]
            if lo_p <= current_price <= hi_p:
                return f"{ratios[i] * 100:.1f}% - {ratios[i + 1] * 100:.1f}%"
    else:  # Downtrend
        # prices ascend as ratio increases: prices[0] = low ... prices[-1] = high
        if current_price <= prices[0]:
            return "Below 0% (new low)"
        if current_price >= prices[-1]:
            return "Beyond 100% (fully retraced)"
        for i in range(len(ratios) - 1):
            lo_p, hi_p = prices[i], prices[i + 1]
            if lo_p <= current_price <= hi_p:
                return f"{ratios[i] * 100:.1f}% - {ratios[i + 1] * 100:.1f}%"

    return "—"


def build_plan(ticker: str, market_regime: Optional[dict] = None) -> TradePlan:
    plan = TradePlan(ticker=ticker)
    try:
        df = fetch_history(ticker)
        current_price = float(df["Close"].iloc[-1])
        current_price, price_is_live = fetch_live_price(ticker, current_price)
        rsi = compute_rsi(df["Close"])
        swing_high, sh_date, swing_low, sl_date, trend, is_extended = detect_last_swing(df)

        # Long-term trend filter: fetched separately with a much longer
        # window purely so SMA_PERIOD has enough trading days to compute,
        # without changing the LOOKBACK_DAYS window used for swing
        # detection above.
        sma200 = None
        try:
            sma_df = fetch_history(ticker, days=LOOKBACK_DAYS + SMA_FETCH_BUFFER_DAYS)
            if len(sma_df) >= SMA_PERIOD:
                sma_val = sma_df["Close"].rolling(SMA_PERIOD).mean().iloc[-1]
                if pd.notna(sma_val):
                    sma200 = float(sma_val)
        except Exception:
            sma200 = None  # non-fatal — trend filter is a bonus check, not a hard requirement

        if swing_high <= swing_low:
            raise ValueError("Could not establish a valid swing range.")

        diff = swing_high - swing_low
        levels = {}

        if trend == "Uptrend":
            # Retracement measured DOWN from the high toward the low.
            for r in FIB_RATIOS:
                levels[r] = swing_high - diff * r
            extension_target = swing_low + diff * FIB_EXTENSION  # == high + 0.618*diff

            entry_high = levels[0.5]
            entry_low = levels[0.618]
            stop_loss = levels[0.786]  # standard stop just beyond the 78.6% retracement
            tp1 = swing_high
            tp2 = extension_target

            if current_price < stop_loss:
                status, status_class, status_label = "❌ INVALIDATED — structure broken (below stop)", "invalid", "Invalidated"
            elif entry_low <= current_price <= entry_high:
                status, status_class, status_label = "✅ IN BUY ZONE — entry active", "buy", "Buy Zone"
            elif entry_high < current_price <= levels[0.382]:
                status, status_class, status_label = "👀 APPROACHING — inside 38.2-50% zone, not yet at entry", "approaching", "38.2-50% Zone"
            elif current_price > levels[0.382]:
                status, status_class, status_label = "⏳ WAITING FOR PULLBACK — price above entry zone", "wait", "Waiting"
            else:  # between stop_loss and entry_low
                status, status_class, status_label = "⚠️ DEEP PULLBACK — inside 61.8-78.6% zone, high risk entry", "risky", "Deep Pullback"

        else:  # Downtrend
            # Retracement measured UP from the low toward the high — used
            # to define resistance zones for short entries / avoiding longs.
            for r in FIB_RATIOS:
                levels[r] = swing_low + diff * r
            extension_target = swing_high - diff * FIB_EXTENSION  # downside extension

            entry_high = levels[0.618]
            entry_low = levels[0.5]
            stop_loss = levels[0.786]  # standard stop just beyond the 78.6% retracement
            tp1 = swing_low
            tp2 = extension_target

            if current_price > stop_loss:
                status, status_class, status_label = "❌ INVALIDATED — price back above resistance stop", "invalid", "Invalidated"
            elif entry_low <= current_price <= entry_high:
                status, status_class, status_label = "🔻 IN SHORT ZONE — no long entries here", "short", "Short Zone"
            elif levels[0.382] <= current_price < entry_low:
                status, status_class, status_label = "👀 APPROACHING — inside 38.2-50% zone, not yet at entry", "approaching", "38.2-50% Zone"
            elif current_price < levels[0.382]:
                status, status_class, status_label = "⏳ DOWNTREND CONTINUING — wait for bounce into zone", "wait", "Waiting"
            else:
                status, status_class, status_label = "⚠️ NEAR RESISTANCE — approaching short entry zone", "risky", "Near Resistance"

        # ── Confirmation checks ──────────────────────────────────────────
        # Sitting inside the Fib zone is a location, not a trigger. These
        # three checks answer: is the pullback happening on light volume
        # (not distribution), has today's candle actually shown a
        # rejection, and does the broader market agree with this leg?

        vol_confirmed = None
        if len(df) >= VOLUME_SMA_PERIOD and "Volume" in df.columns:
            vol_sma = df["Volume"].rolling(VOLUME_SMA_PERIOD).mean().iloc[-1]
            last_vol = df["Volume"].iloc[-1]
            if pd.notna(vol_sma) and vol_sma > 0 and pd.notna(last_vol):
                vol_confirmed = bool(last_vol < vol_sma)

        last_open = float(df["Open"].iloc[-1])
        last_close = float(df["Close"].iloc[-1])
        last_low = float(df["Low"].iloc[-1])
        last_high = float(df["High"].iloc[-1])
        body = abs(last_close - last_open)
        day_range = max(last_high - last_low, 1e-9)
        lower_wick = min(last_open, last_close) - last_low
        upper_wick = last_high - max(last_open, last_close)

        if trend == "Uptrend":
            # Bullish rejection: a green close, or a hammer-style long
            # lower wick (buyers defended the level within the day).
            reversal_confirmed = (last_close > last_open) or (
                lower_wick > body * 1.2 and lower_wick > day_range * 0.33
            )
        else:
            # Bearish rejection: a red close, or a shooting-star long
            # upper wick (sellers defended the level within the day).
            reversal_confirmed = (last_close < last_open) or (
                upper_wick > body * 1.2 and upper_wick > day_range * 0.33
            )

        market_aligned = None
        if market_regime is not None:
            market_aligned = market_regime["bullish"] if trend == "Uptrend" else (not market_regime["bullish"])

        # A single confirmed leg is not a macro trend — cross-check the leg
        # direction against the 200-SMA. If they disagree, don't silently
        # keep issuing a clean Buy/Short signal into the teeth of the
        # long-term trend; relabel it as counter-trend so it's flagged as
        # higher risk rather than presented as a standard entry.
        counter_trend = False
        if sma200 is not None:
            if trend == "Uptrend" and current_price < sma200:
                counter_trend = True
            elif trend == "Downtrend" and current_price > sma200:
                counter_trend = True

        if counter_trend and status_class in ("buy", "short", "approaching"):
            direction = "long-term downtrend" if trend == "Uptrend" else "long-term uptrend"
            status = f"⚠️ COUNTER-TREND — {trend} leg inside a {direction} (price vs 200-SMA); {status}"
            status_class, status_label = "counter", "Counter-Trend"

        # A raw Buy/Short Zone signal is just "price touched the level" —
        # don't hand that out as a clean trigger until volume, candle, and
        # market-regime checks actually back it up. Downgrade to a distinct
        # "pending" state instead of silently keeping the green light on.
        if status_class in ("buy", "short"):
            missing = []
            if vol_confirmed is False:
                missing.append("high-volume pullback (possible distribution)")
            if not reversal_confirmed:
                missing.append("no reversal candle yet")
            if market_aligned is False:
                missing.append("broader market regime disagrees")
            if missing:
                status_class = "pending"
                status_label = "Pending Confirm"
                status = ("🕵️ IN ZONE — AWAITING CONFIRMATION: " + "; ".join(missing) +
                           ". Treat as a watchlist entry, not a trigger, until these clear.")

        conf_bits = []
        conf_bits.append("Vol " + ("Low ✓" if vol_confirmed else ("High ✗" if vol_confirmed is False else "n/a")))
        conf_bits.append("Candle " + ("✓" if reversal_confirmed else "✗"))
        conf_bits.append("Market " + ("Aligned ✓" if market_aligned else ("Against ✗" if market_aligned is False else "n/a")))
        confirmations_note = " · ".join(conf_bits)

        # Surface (rather than hide) legs whose end is a still-forming
        # extreme instead of a fully confirmed zigzag reversal — these
        # Fib levels can still shift as price continues to move.
        if is_extended:
            status += " • Structure still extending — swing extreme not yet confirmed by a reversal, levels may shift"
            status_label = f"{status_label} (Ext.)"

        prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price
        chg_pct = ((current_price - prev_close) / prev_close * 100.0) if prev_close else 0.0
        spark_vals = [float(x) for x in df["Close"].iloc[-20:].tolist()]
        distance_to_entry = abs(current_price - (entry_low + entry_high) / 2.0)

        plan.current_price = current_price
        plan.price_is_live = price_is_live
        plan.chg_pct = chg_pct
        plan.spark = spark_vals
        plan.swing_high, plan.swing_low = swing_high, swing_low
        plan.swing_high_date, plan.swing_low_date = sh_date, sl_date
        plan.extended_leg = is_extended
        plan.rsi = rsi
        plan.sma200 = sma200
        plan.counter_trend = counter_trend
        plan.vol_confirmed = vol_confirmed
        plan.reversal_confirmed = reversal_confirmed
        plan.market_aligned = market_aligned
        plan.confirmations_note = confirmations_note
        plan.trend = trend
        plan.fib_zone = compute_fib_zone(current_price, levels, trend)
        plan.levels = levels
        plan.extension_target = extension_target
        plan.entry_low, plan.entry_high = entry_low, entry_high
        plan.stop_loss = stop_loss
        plan.tp1, plan.tp2 = tp1, tp2
        plan.distance_to_entry = distance_to_entry
        plan.status_label = status_label
        plan.status, plan.status_class = status, status_class

    except Exception as e:
        plan.ok = False
        plan.error = str(e)

    return plan


# ──────────────────────────────────────────────────────────────────────────
# HTML REPORT  (dark dashboard style, sortable table, status pulse, CSV export)
# ──────────────────────────────────────────────────────────────────────────

# status_class -> (display label, dot/accent colour)
STATUS_META = {
    "buy":     ("Buy Zone",       "#26d07c"),
    "pending": ("Pending Confirm","#4fb8f0"),
    "approaching": ("38.2-50% Zone", "#a78bfa"),
    "wait":    ("Waiting",        "#f5c518"),
    "risky":   ("Deep Pullback",  "#f7931e"),
    "invalid": ("Invalidated",    "#ff5470"),
    "short":   ("Short Zone",     "#b06bd6"),
    "counter": ("Counter-Trend",  "#ff9f43"),
    "error":   ("Data Error",     "#8b93a7"),
}


def fmt(x):
    return f"{x:,.2f}"


def make_spark_svg(vals: List[float], color: str = "var(--accent)") -> str:
    if not vals or len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    w, h, pad = 72, 24, 2
    step = (w - 2 * pad) / (len(vals) - 1)
    pts = []
    for i, v in enumerate(vals):
        x = pad + i * step
        y = pad + (1 - (v - lo) / span) * (h - 2 * pad)
        pts.append(f"{x:.1f},{y:.1f}")
    last_x, last_y = pts[-1].split(",")
    return (
        f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
        f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
        f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{last_x}" cy="{last_y}" r="2" fill="{color}"/></svg>'
    )


def render_confirm_chip(label: str, state: Optional[bool]) -> str:
    """state True -> green 'ok', False -> red 'bad', None -> neutral gray."""
    cls = "ok" if state is True else ("bad" if state is False else "neutral")
    mark = "✓" if state is True else ("✗" if state is False else "—")
    return f'<span class="confirm-chip {cls}">{label} {mark}</span>'


def render_card(p: TradePlan) -> str:
    if not p.ok:
        return f"""
        <div class="trade-card error-card" data-symbol="{p.ticker}" data-status="error" data-error="1">
            <b>{p.ticker}</b> &middot; Data error: {p.error}
        </div>"""

    chg_class = "pos" if p.chg_pct >= 0 else "neg"
    chg_sign = "+" if p.chg_pct >= 0 else ""
    trend_up = p.trend == "Uptrend"
    trend_chip_class = "chip-up" if trend_up else "chip-down"
    trend_arrow_cls = "up" if trend_up else "down"
    trend_arrow = "▲" if trend_up else "▼"
    _, status_color = STATUS_META.get(p.status_class, ("", "#6B7280"))
    spark_svg = make_spark_svg(p.spark, color=status_color)

    if p.rsi is None:
        rsi_text = "RSI —"
    else:
        rsi_text = f"RSI {p.rsi:.1f}"

    live_color = "#22C55E" if p.price_is_live else "#8B93A7"
    live_label = "Live" if p.price_is_live else "Delayed (Close)"

    # Volume chip: True (below-avg volume) is the healthy case -> "ok";
    # None means not enough history to judge -> neutral.
    volume_chip = render_confirm_chip("Volume", None if p.vol_confirmed is None else p.vol_confirmed)
    candle_chip = render_confirm_chip("Candle", p.reversal_confirmed)
    market_chip = render_confirm_chip("Market", p.market_aligned)

    return f"""
    <div class="trade-card" data-symbol="{p.ticker}" data-status="{p.status_class}"
        data-ltp="{p.current_price}" data-chg="{p.chg_pct}" data-rsi="{p.rsi if p.rsi is not None else ''}"
        data-trend="{p.trend}" data-fibzone="{p.fib_zone}" data-entrylow="{p.entry_low}" data-entryhigh="{p.entry_high}"
        data-stop="{p.stop_loss}" data-tp1="{p.tp1}" data-tp2="{p.tp2}" data-swinghigh="{p.swing_high}" data-swinglow="{p.swing_low}"
        data-distance="{p.distance_to_entry}">

        <div class="card-block">
            <div class="card-symbol-row">
                <span class="card-symbol">{p.ticker}</span>
                <span class="trend-arrow {trend_arrow_cls}">{trend_arrow}</span>
                <span class="chip {trend_chip_class}">{p.trend}</span>
            </div>
            <div class="card-price-row">
                <span class="card-ltp">{fmt(p.current_price)}</span>
                <span class="{chg_class}" style="font-weight:600;font-size:12.5px;">{chg_sign}{p.chg_pct:.2f}%</span>
            </div>
            <div class="card-sub" style="color:{live_color}">{live_label}</div>
            <div class="card-rsi">{rsi_text}</div>
        </div>

        <div class="card-block">
            <span class="fib-zone-label">FIB Zone: {p.fib_zone}</span>
            <div class="fib-grid">
                <span class="fib-label">Entry Zone</span><span class="fib-value entry">{fmt(p.entry_low)} – {fmt(p.entry_high)}</span>
                <span class="fib-label">Stop-Loss</span><span class="fib-value stop">{fmt(p.stop_loss)}</span>
                <span class="fib-label">TP1</span><span class="fib-value tp">{fmt(p.tp1)}</span>
                <span class="fib-label">TP2 (Ext.)</span><span class="fib-value tp">{fmt(p.tp2)}</span>
            </div>
            <div class="swing-dates">
                <span><span class="ico">📅</span>High {fmt(p.swing_high)} &middot; {p.swing_high_date}</span>
                <span><span class="ico">📅</span>Low {fmt(p.swing_low)} &middot; {p.swing_low_date}</span>
            </div>
        </div>

        <div class="card-block status-block">
            <span class="status-badge {p.status_class}" style="background:{status_color}1a;color:{status_color};border:1px solid {status_color}55">{p.status_label}</span>
            <div class="confirm-chips">{volume_chip}{candle_chip}{market_chip}</div>
            {spark_svg}
        </div>
    </div>"""


def render_pulse_chip(status_class: str, count: int, unit_label: str = "stock(s)") -> str:
    label, color = STATUS_META[status_class]
    icon = {"buy": "✅", "pending": "🕵️", "approaching": "👀", "wait": "⏳", "risky": "⚠️", "invalid": "❌", "short": "🔻", "counter": "🚧", "error": "•"}[status_class]
    return f"""
    <div class="tick-chip" data-status="{status_class}" tabindex="0" role="button" aria-pressed="false"
         style="border-color:{color}55">
        <div class="tick-row1"><span class="tick-icon">{icon}</span><span class="tick-sector">{label}</span></div>
        <div class="tick-row2"><span class="tick-rsi" style="color:{color}">{count}</span><span class="tick-count">{unit_label}</span></div>
        <div class="tick-bar"><span style="width:100%;background:{color}"></span></div>
    </div>"""


def render_panel(panel_id: str, plans: List[TradePlan], unit_label: str = "stock(s)", noun: str = "stocks") -> str:
    """
    Builds the quick-filter toolbar + status pulse strip + sortable table
    for one tab. `panel_id` must be unique ("stocks" / "indices") since
    every element id below is namespaced with it so two panels can coexist
    on the same page and be driven independently by the JS below.
    """
    ok_plans = [p for p in plans if p.ok]
    error_count = len(plans) - len(ok_plans)

    order = ["buy", "pending", "approaching", "risky", "counter", "wait", "short", "invalid"]
    counts = {s: sum(1 for p in ok_plans if p.status_class == s) for s in order}
    pulse_chips = "".join(render_pulse_chip(s, c, unit_label) for s, c in counts.items() if c > 0)
    if error_count:
        pulse_chips += render_pulse_chip("error", error_count, unit_label)

    cards = "\n".join(render_card(p) for p in plans)

    return f"""
    <div class="hdr-bottom">
        <div class="toolbar-left">
            <label for="sortBy-{panel_id}">Sort by</label>
            <select id="sortBy-{panel_id}">
                <option value="none">Default order</option>
                <option value="buyfirst">Buy Zone first</option>
                <option value="nearest">Nearest to Entry</option>
                <option value="gainers">Top Gainers</option>
                <option value="losers">Top Losers</option>
                <option value="symbol">Symbol (A–Z)</option>
                <option value="rsi">RSI (high → low)</option>
            </select>
        </div>
        <div class="toolbar-right">
            <button class="action-btn" id="exportBtn-{panel_id}">&#8681; Export CSV</button>
            <span class="user-pill"><span class="user-dot"></span>Fibonacci Scanner &middot; {LOOKBACK_DAYS}d lookback</span>
        </div>
    </div>

    <div class="panel">
        <div class="ticker-strip" role="group" aria-label="Status pulse">
            <div class="ticker-label">Status pulse
                <span class="ticker-hint">&middot; click a status to filter the cards below</span>
            </div>
            <div class="ticker-grid" id="statusGrid-{panel_id}">{pulse_chips}</div>
        </div>

        <div class="card-grid" id="cardGrid-{panel_id}">
            {cards}
            <div class="no-status-match empty" style="display:none;">No {noun} match this status. Click the chip again to clear the filter.</div>
        </div>

        <footer class="summary-bar">
            <span>{len(ok_plans)} {noun} analyzed{f' &middot; {error_count} data error(s)' if error_count else ''}</span>
            <span>{counts.get('pending', 0)} Pending Confirm &middot; {counts.get('wait', 0) + counts.get('invalid', 0)} No Setup</span>
        </footer>
    </div>"""


def render_html(stock_plans: List[TradePlan], index_plans: List[TradePlan]) -> str:
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stock_ok = [p for p in stock_plans if p.ok]
    index_ok = [p for p in index_plans if p.ok]

    stock_buy_count = sum(1 for p in stock_ok if p.status_class in ("buy", "short"))

    stocks_panel_html = render_panel("stocks", stock_plans, unit_label="stock(s)", noun="stocks")
    indices_panel_html = render_panel("indices", index_plans, unit_label="index(es)", noun="indices")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fibonacci Swing Trade Plan</title>
<style>
    :root {{
        --bg: #0B0E14; --panel: #131722; --panel2: #1A1F2E; --border: #262C3D;
        --text: #E6E9F0; --muted: #8B93A7; --accent: #5B9EFF;
        --pos: #22C55E; --neg: #F87171; --warn: #FACC15;
        --entry-bg: rgba(34,197,94,0.12); --stop-bg: rgba(248,113,113,0.12); --tp-bg: rgba(91,158,255,0.14);
    }}
    * {{ box-sizing: border-box; }}
    body {{
        background: var(--bg); color: var(--text);
        font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
        margin: 0; padding: 28px;
    }}
    .wrap {{ max-width: 1280px; margin: 0 auto; }}

    /* ── Row 1: Title bar ─────────────────────────────────────────── */
    .hdr-title-row {{
        display: flex; align-items: baseline; justify-content: space-between;
        gap: 12px; flex-wrap: wrap; padding: 2px 2px 12px;
    }}
    h1 {{ font-size: 20px; margin: 0; font-weight: 800; letter-spacing: -0.01em; color: var(--text); }}
    .hdr-meta {{ text-align: right; font-size: 12px; color: var(--muted); line-height: 1.6; }}
    .hdr-meta b {{ color: var(--text); font-weight: 700; }}
    .count {{ color: var(--accent); font-weight: 700; }}

    /* ── Row 2: Mode & lookback strip ─────────────────────────────── */
    .mode-strip {{
        display: flex; flex-wrap: wrap; align-items: center; gap: 18px;
        background: var(--panel2); border: 1px solid var(--border);
        border-radius: 8px; padding: 8px 16px; margin-bottom: 10px;
        font-size: 12px; color: #A8B0C4;
    }}
    .mode-item {{ display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }}
    .mode-item .ico {{ font-size: 12px; opacity: 0.8; }}
    .mode-item b {{ color: var(--text); font-weight: 700; }}

    /* ── Row 3: Filter ribbon ─────────────────────────────────────── */
    .criteria-chips {{
        display: flex; flex-wrap: wrap; gap: 6px;
        margin-bottom: 16px;
    }}
    .crit-chip {{
        font-size: 11px; font-weight: 600; letter-spacing: 0.01em;
        padding: 5px 11px; border-radius: 999px; white-space: nowrap;
        background: var(--panel2); border: 1px solid var(--border); color: #A8B0C4;
    }}
    .crit-chip b {{ color: var(--text); font-weight: 700; }}
    .crit-chip.band {{
        color: #FFFFFF; font-weight: 700; border: none;
        background: linear-gradient(90deg, var(--neg) 0%, var(--warn) 50%, var(--pos) 100%);
    }}
    .crit-chip.tf {{ color: var(--accent); border-color: rgba(91,158,255,0.35); background: rgba(91,158,255,0.12); }}

    .hdr-bottom {{
        display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;
        background: var(--panel); border: 1px solid var(--border);
        border-radius: 10px; padding: 10px 18px; margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }}
    .toolbar-left {{ display: flex; align-items: center; gap: 10px; }}
    .toolbar-left label {{ font-size: 12px; color: var(--muted); }}
    .toolbar-left select {{
        background: var(--panel); color: var(--text); border: 1px solid var(--border);
        border-radius: 8px; padding: 6px 10px; font-size: 13px; cursor: pointer;
    }}
    .toolbar-right {{ display: flex; align-items: center; gap: 8px; }}
    .action-btn {{
        display: inline-flex; align-items: center; gap: 6px;
        background: var(--panel); color: var(--text); border: 1px solid var(--border);
        border-radius: 8px; padding: 6px 12px; font-size: 12.5px; font-weight: 600;
        cursor: pointer; transition: border-color 0.15s, color 0.15s;
    }}
    .action-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
    .user-pill {{
        display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted);
        background: var(--panel2); border: 1px solid var(--border); border-radius: 999px; padding: 5px 12px 5px 6px;
    }}
    .user-dot {{ width: 7px; height: 7px; border-radius: 50%; background: var(--pos); box-shadow: 0 0 0 3px rgba(34,197,94,0.22); }}

    .tab-row {{ display: flex; gap: 8px; margin-bottom: 14px; }}
    .tab-btn {{
        display: inline-flex; align-items: center; gap: 8px;
        background: var(--panel); color: var(--muted); border: 1px solid var(--border);
        border-radius: 10px; padding: 9px 16px; font-size: 13px; font-weight: 700;
        cursor: pointer; transition: border-color 0.15s, color 0.15s, background 0.15s;
    }}
    .tab-btn:hover {{ border-color: var(--accent); color: var(--text); }}
    .tab-btn.active {{
        color: #FFFFFF; background: var(--accent); border-color: var(--accent);
    }}
    .tab-count {{
        font-size: 11px; font-weight: 700; background: rgba(0,0,0,0.08);
        border-radius: 999px; padding: 1px 8px;
    }}
    .tab-btn:not(.active) .tab-count {{ background: var(--panel2); color: var(--text); }}

    .panel {{ display: flex; flex-direction: column; gap: 14px; }}

    .ticker-strip {{
        background: var(--panel); border: 1px solid var(--border);
        border-radius: 10px; padding: 12px 18px 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }}
    .ticker-label {{
        font-size: 10px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.08em; color: var(--muted); margin-bottom: 10px;
    }}
    .ticker-hint {{
        text-transform: none; font-weight: 500; letter-spacing: normal;
        color: var(--muted); opacity: 0.85; font-size: 10px;
    }}
    .ticker-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 8px;
    }}
    .tick-chip {{
        display: flex; flex-direction: column; gap: 4px;
        background: var(--panel2); border: 1px solid var(--border);
        border-radius: 8px; padding: 8px 10px;
        cursor: pointer; user-select: none;
        transition: opacity 0.15s, border-color 0.15s, box-shadow 0.15s, transform 0.1s;
    }}
    .tick-chip:hover {{ border-color: var(--accent); }}
    .tick-chip:active {{ transform: scale(0.98); }}
    .tick-chip.active {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent);
        background: rgba(91,158,255,0.12);
    }}
    .ticker-grid.filtering .tick-chip:not(.active) {{ opacity: 0.4; }}
    .tick-row1 {{ display: flex; align-items: center; gap: 6px; }}
    .tick-row2 {{ display: flex; align-items: baseline; justify-content: space-between; gap: 6px; }}
    .tick-icon {{ font-size: 12px; }}
    .tick-sector {{ font-size: 11px; font-weight: 500; color: var(--muted);
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .tick-rsi {{ font-size: 16px; font-weight: 700; font-stretch: condensed; letter-spacing: -0.02em; }}
    .tick-count {{ font-size: 9.5px; color: var(--muted); background: var(--panel);
        border-radius: 999px; padding: 1px 6px; white-space: nowrap; }}
    .tick-bar {{ height: 3px; border-radius: 999px; background: var(--panel); overflow: hidden; }}
    .tick-bar span {{ display: block; height: 100%; border-radius: 999px; }}

    /* ── Card grid ─────────────────────────────────────────────────── */
    .card-grid {{ display: flex; flex-direction: column; gap: 8px; }}
    .trade-card {{
        display: grid;
        grid-template-columns: 1.3fr 2.3fr 1.2fr;
        gap: 0; align-items: stretch;
        background: var(--panel); border: 1px solid var(--border);
        border-radius: 10px; overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        transition: border-color 0.15s, box-shadow 0.15s;
    }}
    .trade-card:hover {{ border-color: #3A4258; box-shadow: 0 2px 10px rgba(0,0,0,0.35); }}
    .trade-card.error-card {{
        grid-template-columns: 1fr; padding: 14px 18px; color: var(--neg); font-size: 13px;
    }}
    .card-block {{ padding: 12px 16px; display: flex; flex-direction: column; justify-content: center; gap: 6px; }}
    .card-block + .card-block {{ border-left: 1px solid var(--border); }}

    /* Left block */
    .card-symbol-row {{ display: flex; align-items: center; gap: 7px; }}
    .card-symbol {{ font-weight: 700; font-size: 14.5px; letter-spacing: -0.01em; }}
    .trend-arrow {{ font-size: 12px; }}
    .trend-arrow.up {{ color: var(--pos); }}
    .trend-arrow.down {{ color: var(--neg); }}
    .card-price-row {{ display: flex; align-items: baseline; gap: 8px; }}
    .card-ltp {{ font-size: 15px; font-weight: 700; font-variant-numeric: tabular-nums; }}
    .card-live-tag {{ font-size: 9.5px; font-weight: 600; }}
    .card-sub {{ font-size: 10.5px; color: var(--muted); }}
    .card-rsi {{ font-size: 11px; color: var(--muted); }}

    /* Middle block: fib mini-grid */
    .fib-zone-label {{ font-size: 10.5px; color: var(--accent); font-weight: 700;
        background: rgba(91,158,255,0.12); border: 1px solid rgba(91,158,255,0.3);
        border-radius: 6px; padding: 2px 8px; display: inline-block; width: fit-content; }}
    .fib-grid {{ display: grid; grid-template-columns: auto 1fr; column-gap: 10px; row-gap: 4px; font-size: 12px; margin-top: 2px; }}
    .fib-label {{ color: var(--muted); text-align: left; white-space: nowrap; }}
    .fib-value {{ text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }}
    .fib-value.entry {{ background: var(--entry-bg); color: #4ADE80; border-radius: 4px; padding: 1px 6px; }}
    .fib-value.stop {{ background: var(--stop-bg); color: #FCA5A5; border-radius: 4px; padding: 1px 6px; }}
    .fib-value.tp {{ background: var(--tp-bg); color: #93C5FD; border-radius: 4px; padding: 1px 6px; }}
    .swing-dates {{ display: flex; gap: 14px; margin-top: 4px; font-size: 10px; color: var(--muted); }}
    .swing-dates .ico {{ margin-right: 3px; }}

    /* Right block: status */
    .card-block.status-block {{ align-items: flex-end; text-align: right; gap: 8px; }}
    .status-badge {{ font-weight: 700; padding: 4px 11px; border-radius: 999px; font-size: 11.5px; white-space: nowrap; }}
    .confirm-chips {{ display: flex; flex-wrap: wrap; gap: 4px; justify-content: flex-end; }}
    .confirm-chip {{ font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 999px; white-space: nowrap; }}
    .confirm-chip.ok {{ background: rgba(34,197,94,0.16); color: #4ADE80; }}
    .confirm-chip.bad {{ background: rgba(248,113,113,0.16); color: #FCA5A5; }}
    .confirm-chip.warn {{ background: rgba(250,204,21,0.18); color: #FDE68A; }}
    .confirm-chip.neutral {{ background: var(--panel2); color: var(--muted); }}
    .spark {{ display: block; margin-top: 2px; }}

    .sym {{ font-weight: 600; }}
    .sub-date {{ font-size: 10px; color: var(--muted); font-weight: 400; }}
    .pos {{ color: var(--pos); }}
    .neg {{ color: var(--neg); }}
    .chip {{ padding: 2px 9px; border-radius: 999px; font-size: 11px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.03em; }}
    .chip-up {{ background: rgba(34,197,94,0.14); color: var(--pos); }}
    .chip-down {{ background: rgba(248,113,113,0.14); color: var(--neg); }}
    .empty {{ text-align: center; color: var(--muted); padding: 32px; }}

    footer.summary-bar {{
        display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;
        margin-top: 14px; padding: 10px 18px; background: var(--panel);
        border: 1px solid var(--border); border-radius: 10px; font-size: 12px; color: var(--muted);
    }}

    @media (max-width: 860px) {{
        .trade-card {{ grid-template-columns: 1fr; }}
        .card-block + .card-block {{ border-left: none; border-top: 1px solid var(--border); }}
        .card-block.status-block {{ align-items: flex-start; text-align: left; }}
        .confirm-chips {{ justify-content: flex-start; }}
        .hdr-title-row {{ flex-direction: column; align-items: flex-start; }}
        .hdr-meta {{ text-align: left; }}
    }}
</style>
</head>
<body>
<div class="wrap">
    <!-- Row 1 — Title bar -->
    <div class="hdr-title-row">
        <h1>📐 Fibonacci Swing Trade Plan</h1>
        <div class="hdr-meta" id="subtitle">
            Generated: <b>{generated}</b><br>
            Universe: <b>{len(stock_ok)} Stocks</b> | <b>{len(index_ok)} Indices</b> &middot; <span class="count">{stock_buy_count}</span> in Buy/Short Zone
        </div>
    </div>

    <!-- Row 2 — Mode & lookback context -->
    <div class="mode-strip">
        <span class="mode-item"><span class="ico">⏱</span>Lookback: <b>{LOOKBACK_DAYS}d</b></span>
        <span class="mode-item"><span class="ico">🔀</span>Swing Mode: <b>{SWING_MODE.title()}</b></span>
        <span class="mode-item"><span class="ico">📈</span>Trend Filter: <b>{SMA_PERIOD}-SMA</b></span>
        <span class="mode-item"><span class="ico">📐</span>Zigzag Threshold: <b>{ZIGZAG_PCT:g}%</b></span>
    </div>

    <!-- Row 3 — Filter ribbon -->
    <div class="criteria-chips" title="Standard Fibonacci swing-trade rules: retracement levels measured off the most recent detected swing leg, entry inside the 50-61.8% pocket, stop beyond 78.6%, targets at the prior swing extreme and the 161.8% extension.">
        <span class="crit-chip band">Entry Zone: 50&ndash;61.8%</span>
        <span class="crit-chip">Retracements: <b>23.6 / 38.2 / 50 / 61.8 / 78.6</b></span>
        <span class="crit-chip">Stop: <b>78.6%</b></span>
        <span class="crit-chip">Extension: <b>161.8%</b></span>
    </div>

    <div class="tab-row" role="tablist" aria-label="Ticker universe">
        <button class="tab-btn active" id="tabBtn-stocks" role="tab" aria-selected="true" onclick="showTab('stocks')">
            📈 Stocks <span class="tab-count">{len(stock_ok)}</span>
        </button>
        <button class="tab-btn" id="tabBtn-indices" role="tab" aria-selected="false" onclick="showTab('indices')">
            🏛️ Indices <span class="tab-count">{len(index_ok)}</span>
        </button>
    </div>

    <div id="panel-stocks">
        {stocks_panel_html}
    </div>
    <div id="panel-indices" style="display:none;">
        {indices_panel_html}
    </div>
</div>
<script>
    function initPanel(panelId) {{
        const grid = document.getElementById('cardGrid-' + panelId);
        const statusGrid = document.getElementById('statusGrid-' + panelId);
        const noMatchCard = grid.querySelector('.no-status-match');
        const sortBy = document.getElementById('sortBy-' + panelId);
        const exportBtn = document.getElementById('exportBtn-' + panelId);
        let activeStatus = null;

        function dataCards() {{
            return Array.from(grid.querySelectorAll('.trade-card')).filter(c => c.dataset.symbol);
        }}

        function cellValue(card, key) {{
            const map = {{
                symbol: () => card.dataset.symbol,
                ltp: () => parseFloat(card.dataset.ltp),
                chg: () => parseFloat(card.dataset.chg),
                rsi: () => parseFloat(card.dataset.rsi),
                trend: () => card.dataset.trend,
                fibzone: () => card.dataset.fibzone,
                swinghigh: () => parseFloat(card.dataset.swinghigh),
                swinglow: () => parseFloat(card.dataset.swinglow),
                entrylow: () => parseFloat(card.dataset.entrylow),
                stop: () => parseFloat(card.dataset.stop),
                tp1: () => parseFloat(card.dataset.tp1),
                tp2: () => parseFloat(card.dataset.tp2),
                distance: () => parseFloat(card.dataset.distance),
            }};
            const v = map[key] ? map[key]() : '';
            return (typeof v === 'string') ? v : (isNaN(v) ? -Infinity : v);
        }}

        function sortCards(key, dir) {{
            const cards = dataCards();
            if (!cards.length) return;
            cards.sort((a, b) => {{
                const av = cellValue(a, key), bv = cellValue(b, key);
                if (av < bv) return -1 * dir;
                if (av > bv) return 1 * dir;
                return 0;
            }});
            cards.forEach(c => grid.appendChild(c));
            grid.appendChild(noMatchCard);
        }}

        function applyStatusFilter(status) {{
            const cards = dataCards();
            let visible = 0;
            cards.forEach(c => {{
                const match = c.dataset.status === status;
                c.style.display = match ? '' : 'none';
                if (match) visible++;
            }});
            noMatchCard.style.display = (visible === 0) ? '' : 'none';
        }}

        function clearStatusFilter() {{
            dataCards().forEach(c => {{ c.style.display = ''; }});
            noMatchCard.style.display = 'none';
        }}

        statusGrid.querySelectorAll('.tick-chip').forEach(chip => {{
            const status = chip.dataset.status;
            const toggle = () => {{
                if (activeStatus === status) {{
                    activeStatus = null;
                    statusGrid.classList.remove('filtering');
                    chip.classList.remove('active');
                    chip.setAttribute('aria-pressed', 'false');
                    clearStatusFilter();
                }} else {{
                    activeStatus = status;
                    statusGrid.classList.add('filtering');
                    statusGrid.querySelectorAll('.tick-chip').forEach(c => {{
                        const isActive = c === chip;
                        c.classList.toggle('active', isActive);
                        c.setAttribute('aria-pressed', isActive ? 'true' : 'false');
                    }});
                    applyStatusFilter(status);
                }}
            }};
            chip.addEventListener('click', toggle);
            chip.addEventListener('keydown', e => {{
                if (e.key === 'Enter' || e.key === ' ') {{ e.preventDefault(); toggle(); }}
            }});
        }});

        document.addEventListener('keydown', e => {{
            if (e.key === 'Escape' && activeStatus) {{
                const activeChip = statusGrid.querySelector('.tick-chip.active');
                if (activeChip) activeChip.click();
            }}
        }});

        sortBy.addEventListener('change', () => {{
            const value = sortBy.value;
            if (value === 'none') return;
            if (value === 'buyfirst') {{
                const cards = dataCards();
                cards.sort((a, b) => (a.dataset.status === 'buy' ? -1 : 1) - (b.dataset.status === 'buy' ? -1 : 1));
                cards.forEach(c => grid.appendChild(c));
                grid.appendChild(noMatchCard);
            }} else if (value === 'nearest') {{
                sortCards('distance', 1);
            }} else if (value === 'gainers') {{
                sortCards('chg', -1);
            }} else if (value === 'losers') {{
                sortCards('chg', 1);
            }} else if (value === 'symbol') {{
                sortCards('symbol', 1);
            }} else if (value === 'rsi') {{
                sortCards('rsi', -1);
            }}
        }});

        exportBtn.addEventListener('click', () => {{
            const fields = ['symbol','ltp','chg','rsi','trend','fibzone','swinghigh','swinglow','entrylow','entryhigh','stop','tp1','tp2','status'];
            const cards = dataCards().filter(c => c.style.display !== 'none');
            if (!cards.length) return;
            const lines = [fields.join(',')];
            cards.forEach(c => {{
                const row = fields.map(f => f === 'status' ? c.dataset.status : (c.dataset[f] ?? ''));
                lines.push(row.map(v => '"' + String(v).replace(/"/g, '""') + '"').join(','));
            }});
            const blob = new Blob([lines.join('\\n')], {{ type: 'text/csv' }});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'fibonacci_trade_plan_' + panelId + '.csv';
            a.click();
            URL.revokeObjectURL(a.href);
        }});
    }}

    initPanel('stocks');
    initPanel('indices');

    function showTab(tab) {{
        document.getElementById('panel-stocks').style.display = (tab === 'stocks') ? '' : 'none';
        document.getElementById('panel-indices').style.display = (tab === 'indices') ? '' : 'none';
        document.getElementById('tabBtn-stocks').classList.toggle('active', tab === 'stocks');
        document.getElementById('tabBtn-indices').classList.toggle('active', tab === 'indices');
    }}
</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────

def main():
    print(f"Fibonacci Swing Trade Plan — {len(TICKERS)} tickers, {LOOKBACK_DAYS}d lookback, "
          f"swing_mode={SWING_MODE}")
    print(f"Output will be written to: {OUTPUT_HTML}\n")

    print(f"Checking market regime ({MARKET_INDEX_TICKER} vs {MARKET_SMA_PERIOD}-SMA)...")
    market_regime = get_market_regime()
    if market_regime is None:
        print("  Could not fetch index data — market-regime filter disabled for this run.\n")
    else:
        state = "BULLISH (above SMA)" if market_regime["bullish"] else "BEARISH (below SMA)"
        print(f"  {MARKET_INDEX_TICKER}: {market_regime['price']:.2f} vs SMA {market_regime['sma']:.2f} -> {state}\n")

    plans = []
    ok_count = 0
    for i, t in enumerate(TICKERS, 1):
        plan = build_plan(t, market_regime=market_regime)
        plans.append(plan)
        if plan.ok:
            ok_count += 1
            print(f"[{i:>3}/{len(TICKERS)}] {t:<16} OK   trend={plan.trend:<9} "
                  f"LTP={plan.current_price:>10.2f}  status={plan.status_label}")
        else:
            print(f"[{i:>3}/{len(TICKERS)}] {t:<16} FAIL {plan.error}")

    print(f"\nScanning {len(INDEX_TICKERS)} major indices...")
    index_plans = []
    index_ok_count = 0
    for i, t in enumerate(INDEX_TICKERS, 1):
        plan = build_plan(t, market_regime=market_regime)
        index_plans.append(plan)
        if plan.ok:
            index_ok_count += 1
            print(f"[{i:>3}/{len(INDEX_TICKERS)}] {t:<16} OK   trend={plan.trend:<9} "
                  f"LTP={plan.current_price:>10.2f}  status={plan.status_label}")
        else:
            print(f"[{i:>3}/{len(INDEX_TICKERS)}] {t:<16} FAIL {plan.error}")

    html = render_html(plans, index_plans)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nDone: {ok_count}/{len(TICKERS)} stocks and {index_ok_count}/{len(INDEX_TICKERS)} indices analyzed successfully.")
    print(f"Report saved to: {OUTPUT_HTML}")
    print("Open it in your browser to view the interactive trade plan.")


if __name__ == "__main__":
    main()
