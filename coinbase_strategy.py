"""
Coinbase Automated Trading Strategy - EMA Golden Cross + RSI + MACD
=====================================================
Parameters:
  Short-term EMA : 20 days
  Long-term EMA  : 50 days
  RSI            : 14 periods
  MACD           : 12 / 26 / 9
  Coinbase Fee   : 0.6% (Taker fee, per side)

Strategy Rules:
  Buy Conditions:
    1. EMA20 crosses above EMA50 (Golden Cross)
    2. RSI < 70 (Not overbought)
    3. MACD histogram > 0 (Upward momentum)

  Sell Conditions (In priority):
    A. EMA20 crosses below EMA50 (Death Cross) -> Sell All
    B. Price drops below EMA50 * (1 - 3%)      -> Sell All
    C. Price drops below EMA20 * (1 - 3%)      -> Sell 50%

  Re-entry Conditions (After partial sell from C):
    - EMA20 & EMA50 still upward (slope > 0)
    - Price regains EMA20 or bounces off EMA50
    - RSI < 70, MACD histogram > 0
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
import json
import os
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
import sys
from coinbase.rest import RESTClient

# Load environment variables
load_dotenv()

# RL related imports
try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")
DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"
PRODUCT_ID = os.getenv("PRODUCT_ID", "BTC-USD")

# ─────────────────────────────────────────
# 1. Parameters (Entry 20/50, Exit 5/10)
# ─────────────────────────────────────────
ENTRY_SHORT = 20
ENTRY_LONG  = 50
EXIT_SHORT  = 5
EXIT_LONG   = 10

RSI_PERIOD  = 14
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9
FEE_RATE    = 0.006   # Coinbase Taker fee 0.6% per side; Buy+Sell = 1.2%
HALF_SELL_THRESHOLD  = 0.03   # Drop below short-term EMA by 3%
FULL_SELL_THRESHOLD  = 0.03   # Drop below long-term EMA by 3%
INITIAL_CAPITAL = 500      # Initial Capital (USD)

VIRTUAL_STATE_FILE = "virtual_state.json"
TRADE_LOG_FILE     = "trade_log.csv"

# ─────────────────────────────────────────
# 2. Fetch Historical Data (Coinbase Advanced API)
# ─────────────────────────────────────────
def fetch_coinbase_candles(product_id: str = "BTC-USD",
                            granularity: str = "ONE_DAY",
                            days: int = 300,
                            start_date: str = None) -> pd.DataFrame:
    """
    Fetch OHLCV data from Coinbase Advanced Trade API.
    Supports chunked fetching for long durations.
    """
    now_ts = int(datetime.utcnow().timestamp())
    
    if start_date:
        try:
            dt = datetime.strptime(start_date, "%Y-%m-%d")
            requested_start_ts = int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            print(f"Date format error ({start_date}), falling back to days: {days}")
            requested_start_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    else:
        requested_start_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp())

    all_data = []
    current_end = now_ts
    
    # Coinbase API limits around 300 candles per call
    # We fetch in 300-day chunks until requested_start_ts is covered.
    chunk_seconds = 300 * 24 * 3600 # 300 days
    
    while current_end > requested_start_ts:
        current_start = max(requested_start_ts, current_end - chunk_seconds)
        
        url = (f"https://api.coinbase.com/api/v3/brokerage/market/products/"
               f"{product_id}/candles"
               f"?start={current_start}&end={current_end}&granularity={granularity}")

        headers = {"Content-Type": "application/json"}
        resp = requests.get(url, headers=headers, timeout=60)
        
        if resp.status_code != 200:
            print(f"Fetch failed ({current_start} to {current_end}): {resp.text}")
            break
            
        data = resp.json().get("candles", [])
        if not data:
            break
            
        all_data.extend(data)
        # Update end time for next chunk
        current_end = current_start - 1
        
        # Rate limit safety
        time.sleep(0.5)

    if not all_data:
        raise ValueError("No historical data fetched. Check network or date settings.")

    df = pd.DataFrame(all_data, columns=["start", "low", "high", "open", "close", "volume"])
    df["date"]  = pd.to_datetime(df["start"].astype(int), unit="s")
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date","open","high","low","close","volume"]]

# ─────────────────────────────────────────
# 3. Technical Indicators
# ─────────────────────────────────────────
def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = calc_ema(series, fast)
    ema_slow   = calc_ema(series, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Entry Indicators
    df["ema_entry_s"] = calc_ema(df["close"], ENTRY_SHORT)
    df["ema_entry_l"] = calc_ema(df["close"], ENTRY_LONG)
    # Exit Indicators
    df["ema_exit_s"]  = calc_ema(df["close"], EXIT_SHORT)
    df["ema_exit_l"]  = calc_ema(df["close"], EXIT_LONG)
    
    # Extra Indicators (For strategy comparison)
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["ma50"] = df["close"].rolling(window=50).mean()
    
    df["rsi"] = calc_rsi(df["close"], RSI_PERIOD)
    df["macd"], df["macd_signal"], df["macd_hist"] = calc_macd(
        df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    # Entry Crossovers (Dual EMA)
    df["entry_golden_cross"] = (df["ema_entry_s"] > df["ema_entry_l"]) & \
                               (df["ema_entry_s"].shift(1) <= df["ema_entry_l"].shift(1))
    
    # Exit Crossovers (Dual EMA)
    df["exit_death_cross"]   = (df["ema_exit_s"] < df["ema_exit_l"]) & \
                               (df["ema_exit_s"].shift(1) >= df["ema_exit_l"].shift(1))
    
    # Slopes for re-buy judgment
    df["ema_entry_s_slope"] = df["ema_entry_s"].diff()
    df["ema_entry_l_slope"] = df["ema_entry_l"].diff()
    
    return df

# ─────────────────────────────────────────
# 4. Backtest Engine
# ─────────────────────────────────────────
def run_backtest(df: pd.DataFrame) -> dict:
    """Execute Dual EMA strategy backtest"""
    # Redundant indicators already handled externally

    cash        = float(INITIAL_CAPITAL)
    holdings    = 0.0        # BTC holdings
    avg_cost    = 0.0        # Average cost
    half_sold   = False      # Partial sell flag
    trades      = []
    equity_curve = []

    def buy(idx, price, reason, frac=1.0):
        nonlocal cash, holdings, avg_cost, half_sold
        spend = cash * frac
        if spend < 1:
            return
        fee   = spend * FEE_RATE
        qty   = (spend - fee) / price
        holdings += qty
        avg_cost  = price
        cash     -= spend
        half_sold = False
        trades.append({"date": str(df.loc[idx,"date"].date()),
                        "action": "BUY", "price": round(price,2),
                        "qty": round(qty,6), "fee": round(fee,2),
                        "reason": reason})

    def sell(idx, price, reason, frac=1.0):
        nonlocal cash, holdings, half_sold
        qty   = holdings * frac
        if qty < 1e-8:
            return
        gross = qty * price
        fee   = gross * FEE_RATE
        cash += gross - fee
        holdings -= qty
        if frac < 1.0:
            half_sold = True
        trades.append({"date": str(df.loc[idx,"date"].date()),
                        "action": f"SELL {int(frac*100)}%",
                        "price": round(price,2),
                        "qty": round(qty,6), "fee": round(fee,2),
                        "reason": reason})

    for i in range(1, len(df)):
        row   = df.loc[i]
        price = row["close"]
        ema_entry_s = row["ema_entry_s"]
        ema_entry_l = row["ema_entry_l"]
        ema_exit_s  = row["ema_exit_s"]
        ema_exit_l  = row["ema_exit_l"]
        rsi   = row["rsi"]
        hist  = row["macd_hist"]

        equity = cash + holdings * price
        equity_curve.append({"date": str(row["date"].date()), "equity": round(equity, 2),
                              "price": round(price, 2)})

        both_entry_up = row["ema_entry_s_slope"] > 0 and row["ema_entry_l_slope"] > 0

        # ── Sell Logic (Using 5/10 exit signals) ──────────────────────
        if holdings > 1e-8:
            # A: 5/10 Death Cross -> Sell All
            if row["exit_death_cross"]:
                sell(i, price, "5/10 Death Cross (Exit)", frac=1.0)
                half_sold = False
                continue

            # B: Drop below EMA10 * (1 - 3%) -> Sell All (Hard Stop Loss)
            if price < ema_exit_l * (1 - FULL_SELL_THRESHOLD):
                sell(i, price, f"Below EMA10-{int(FULL_SELL_THRESHOLD*100)}%", frac=1.0)
                half_sold = False
                continue

            # C: Price < EMA5 * (1 - 3%) -> Sell Half
            if price < ema_exit_s * (1 - HALF_SELL_THRESHOLD) and not half_sold:
                sell(i, price, f"Below EMA5-{int(HALF_SELL_THRESHOLD*100)}%", frac=0.5)

        # ── Buy Logic (Using 20/50 entry signals) ────────────────────────────
        buy_signal = (row["entry_golden_cross"] and rsi < 70 and hist > 0)

        if buy_signal and holdings < 1e-8:
            buy(i, price, "20/50 Golden Cross (Strong Buy)", frac=1.0)

        # ── Re-buy (After partial sell, rebound conditions) ─────
        rebuy_signal = half_sold and both_entry_up and (price >= ema_entry_s) and rsi < 70 and hist > 0
        if rebuy_signal:
            buy(i, price, "Regained EMA20 (Re-buy)", frac=1.0)

    # Final close at end of backtest
    if holdings > 1e-8:
        last = df.iloc[-1]
        sell(len(df)-1, last["close"], "Backtest end close", frac=1.0)

    final_equity = cash
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    # Calculate Buy & Hold: from the start of the backtest period
    # To be fair, compared to the strategy which also starts after warm-up
    start_price = df.iloc[0]["close"]
    end_price   = df.iloc[-1]["close"]
    buy_hold     = (end_price / start_price - 1) * 100

    # Max Drawdown
    eq_vals = [e["equity"] for e in equity_curve]
    peak, max_dd = INITIAL_CAPITAL, 0.0
    for v in eq_vals:
        if v > peak: peak = v
        dd = (peak - v) / peak
        if dd > max_dd: max_dd = dd

    total_fees = sum(t["fee"] for t in trades)

    return {
        "strategy_name": "Dual EMA (20/50 + 5/10)",
        "start_date": str(df.iloc[0]["date"].date()),
        "end_date": str(df.iloc[-1]["date"].date()),
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return, 2),
        "buy_hold_return_pct": round(buy_hold, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_trades": len(trades),
        "total_fees_paid": round(total_fees, 2),
        "trades": trades,
        "equity_curve": equity_curve,
    }

def run_ma_cross_backtest(df: pd.DataFrame, buffer_pct: float = 0.01) -> dict:
    """Implement 20MA/50MA Crossover strategy (from Sideproject)"""
    # Remove redundant dropna
    cash = float(INITIAL_CAPITAL)
    holdings = 0.0
    trades = []
    equity_curve = []

    for i in range(len(df)):
        row = df.loc[i]
        price = row["close"]
        m20 = row["ma20"]
        m50 = row["ma50"]

        # Buy Signal
        if m20 > m50 and cash > 1:
            fee = cash * FEE_RATE
            qty = (cash - fee) / price
            holdings += qty
            cash = 0
            trades.append({"action": "BUY", "price": price})

        # Sell Signal
        elif holdings > 1e-8 and price < m20 * (1 - buffer_pct):
            if price < m50 * (1 - buffer_pct):
                # Sell All
                rev = holdings * price * (1 - FEE_RATE)
                cash += rev
                holdings = 0
                trades.append({"action": "SELL_ALL", "price": price})
            else:
                # Sell Half
                qty_half = holdings * 0.5
                rev = qty_half * price * (1 - FEE_RATE)
                cash += rev
                holdings -= qty_half
                trades.append({"action": "SELL_HALF", "price": price})

        equity = cash + holdings * price
        equity_curve.append(equity)

    final_equity = cash + holdings * df.iloc[-1]["close"]
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    # Calculate Max Drawdown
    peak, max_dd = INITIAL_CAPITAL, 0.0
    for v in equity_curve:
        if v > peak: peak = v
        dd = (peak - v) / peak
        if dd > max_dd: max_dd = dd

    return {
        "strategy_name": "20/50 MA Cross (User)",
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_trades": len(trades)
    }

def run_grid_backtest(df: pd.DataFrame, grid_count: int = 10, width_pct: float = 0.05) -> dict:
    """Implement Static Grid trading strategy (from Sideproject)"""
    # Remove redundant dropna
    cash = float(INITIAL_CAPITAL)
    holdings = 0.0
    trades_count = 0
    
    current_price = df.iloc[0]["close"]
    center = current_price
    upper = center * (1 + width_pct)
    lower = center * (1 - width_pct)
    grid_step = (upper - lower) / grid_count
    
    levels = [lower + i * grid_step for i in range(grid_count + 1)]
    equity_curve = []

    for i in range(1, len(df)):
        row = df.loc[i]
        high = row["high"]
        low = row["low"]
        close = row["close"]
        
        # Simulating grid triggers (Simplified version)
        for lv in levels:
            # Price drops below grid level -> Buy (Assume allocating 1/N of capital per grid)
            if current_price > lv and low <= lv:
                buy_amt = INITIAL_CAPITAL / grid_count
                if cash >= buy_amt:
                    qty = (buy_amt * (1 - FEE_RATE)) / lv
                    holdings += qty
                    cash -= buy_amt
                    trades_count += 1
            
            # Price rises above grid level -> Sell
            if current_price < lv and high >= lv:
                if holdings > 0:
                    sell_qty = holdings / grid_count # Simplify: sell 1/N of total holdings
                    cash += sell_qty * lv * (1 - FEE_RATE)
                    holdings -= sell_qty
                    trades_count += 1
        
        current_price = close
        equity = cash + holdings * close
        equity_curve.append(equity)

    final_equity = cash + holdings * df.iloc[-1]["close"]
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    peak, max_dd = INITIAL_CAPITAL, 0.0
    for v in equity_curve:
        if v > peak: peak = v
        dd = (peak - v) / peak
        if dd > max_dd: max_dd = dd

    return {
        "strategy_name": "Static Grid (10 grids)",
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_trades": trades_count
    }

# 5. Reinforcement Learning Strategy (Artemis Triple-Barrier RL)
# ─────────────────────────────────────────

RL_STATE_FILE = "rl_state.json"

def get_rl_state():
    if os.path.exists(RL_STATE_FILE):
        with open(RL_STATE_FILE, "r") as f:
            return json.load(f)
    return {"in_position": False, "position_type": 0, "entry_price": 0.0, "tp_price": 0.0, "sl_price": 0.0, "hold_steps": 0, "qty": 0.0}

def save_rl_state(state):
    with open(RL_STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def compute_rl_features_tb(df: pd.DataFrame, state: dict = None) -> np.ndarray:
    """Implement 20 features for Artemis Triple-Barrier RL"""
    prices = df['close'].values.astype(np.float32)
    highs = df['high'].values.astype(np.float32) if 'high' in df else prices * 1.001
    lows = df['low'].values.astype(np.float32) if 'low' in df else prices * 0.999
    volumes = df['volume'].values.astype(np.float32) if 'volume' in df else np.ones_like(prices)
    
    log_ret = np.log(prices[1:] / (prices[:-1] + 1e-9))
    log_ret = np.concatenate([[0.0], log_ret])
    
    # Technical Indicators (Normalized)
    def get_rsi(p):
        delta = np.diff(p, prepend=p[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        ag = pd.Series(gain).ewm(span=14, adjust=False).mean().values
        al = pd.Series(loss).ewm(span=14, adjust=False).mean().values
        rs = ag / (al + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return (rsi - 50) / 50

    def get_macd(p):
        s = pd.Series(p)
        f = s.ewm(span=12, adjust=False).mean().values
        sl = s.ewm(span=26, adjust=False).mean().values
        m = f - sl
        sig = pd.Series(m).ewm(span=9, adjust=False).mean().values
        h = m - sig
        return m / (p + 1e-9), sig / (p + 1e-9), h / (p + 1e-9)

    def get_atr(h, l, c):
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        a = pd.Series(tr).ewm(span=14, adjust=False).mean().values
        return a / (c + 1e-9)

    rsi = get_rsi(prices)
    m, ms, mh = get_macd(prices)
    atr = get_atr(highs, lows, prices)
    vol20 = pd.Series(log_ret).rolling(20, min_periods=1).std(ddof=0).values.astype(np.float32)
    mom5 = np.concatenate([[0.0]*5, (prices[5:] - prices[:-5]) / (prices[:-5] + 1e-9)])
    
    t = -1
    price = prices[t]
    
    pos_val = 0.0
    unrealised_pnl, dist_tp, dist_sl, steps_rem = 0.0, 0.0, 0.0, 0.0
    
    if state and state.get("in_position"):
        pos_val = 1.0 if state["position_type"] == 1 else -1.0
        entry = state["entry_price"]
        unrealised_pnl = (price - entry) / entry * pos_val
        dist_tp = abs(state["tp_price"] - price) / entry
        dist_sl = abs(price - state["sl_price"]) / entry
        steps_rem = max(0, 50 - state["hold_steps"]) / 50.0

    obs = np.array([
        log_ret[t], log_ret[max(t-1, 0)], log_ret[max(t-2, 0)],
        rsi[t], m[t], ms[t], mh[t], 0.0, # Bollinger placeholder
        atr[t], np.clip(mom5[t], -0.2, 0.2) / 0.2, 0.0, 0.0, # Mom10, VZ placeholders
        np.clip(vol20[t] * 100, 0, 5) / 5, 0.0, 0.0, # Portfolio growth, DD placeholders
        pos_val, np.clip(unrealised_pnl * 10, -1, 1),
        np.clip(dist_tp * 10, 0, 1), np.clip(dist_sl * 10, 0, 1), steps_rem
    ], dtype=np.float32)
    
    return obs.reshape(1, -1), atr[t] * price

def run_rl_backtest_tb(df: pd.DataFrame, model_path: str = "models/artemis_tb_v1.zip") -> dict:
    """Execute Artemis Triple-Barrier RL model simulation backtest"""
    if not RL_AVAILABLE or not os.path.exists(model_path):
        return {"strategy_name": "Artemis Triple-Barrier", "error": "Model not found"}

    model = PPO.load(model_path)
    initial_cap = 10000.0
    cash, holdings, equity_curve = initial_cap, 0.0, []
    state = {"in_position": False, "position_type": 0, "entry_price": 0.0, "tp_price": 0.0, "sl_price": 0.0, "hold_steps": 0, "qty": 0.0}
    trades_count = 0

    for i in range(20, len(df)):
        df_slice = df.iloc[:i+1]
        price = df_slice.iloc[-1]["close"]
        
        # 1. Check Barriers
        if state["in_position"]:
            hit_tp = price >= state["tp_price"] if state["position_type"] == 1 else price <= state["tp_price"]
            hit_sl = price <= state["sl_price"] if state["position_type"] == 1 else price >= state["sl_price"]
            if hit_tp or hit_sl or state["hold_steps"] >= 50:
                revenue = state["qty"] * price * (1 - FEE_RATE)
                cash += revenue
                state["qty"] = 0.0
                state["in_position"] = False
                trades_count += 1
            else:
                state["hold_steps"] += 1
        
        # 2. Predict
        obs, cur_atr = compute_rl_features_tb(df_slice, state)
        action, _ = model.predict(obs, deterministic=True)
        action = int(action[0])
        
        if action == 1 and not state["in_position"]: # Long
            fee = cash * FEE_RATE
            state["qty"] = (cash - fee) / price
            state["entry_price"], state["in_position"], state["position_type"], state["hold_steps"] = price, True, 1, 0
            dist = max(cur_atr * 2.0, price * 0.01)
            state["tp_price"], state["sl_price"] = price + dist, price - dist
            cash = 0.0
        elif action == 0 and state["in_position"]: # Exit
            cash += state["qty"] * price * (1 - FEE_RATE)
            state["qty"] = 0.0
            state["in_position"] = False
            trades_count += 1
            
        equity_curve.append(cash + state["qty"] * price)

    final = equity_curve[-1] if equity_curve else initial_cap
    max_dd = 0
    peak = initial_cap
    for v in equity_curve:
        if v > peak: peak = v
        dd = (peak - v) / peak
        if dd > max_dd: max_dd = dd

    return {"strategy_name": "Artemis Triple-Barrier", "total_return_pct": round((final/initial_cap-1)*100, 2), "max_drawdown_pct": round(max_dd*100, 2), "total_trades": trades_count}

def get_rl_signal_tb(product_id: str, model_path: str = "models/artemis_tb_v1.zip"):
    """Get Triple-Barrier RL signal"""
    if not RL_AVAILABLE or not os.path.exists(model_path):
        return {"signal": "HOLD", "reason": "RL Model Unavailable"}
        
    df = fetch_coinbase_candles(product_id, days=100)
    state = get_rl_state()
    obs, cur_atr = compute_rl_features_tb(df, state)
    
    model = PPO.load(model_path)
    action, _ = model.predict(obs, deterministic=True)
    action = int(action[0])
    price = df.iloc[-1]["close"]
    
    # Check Barriers
    if state["in_position"]:
        hit_tp = price >= state["tp_price"]
        hit_sl = price <= state["sl_price"]
        if hit_tp: return {"signal": "SELL_ALL", "reason": "RL: Hit Take Profit"}
        if hit_sl: return {"signal": "SELL_ALL", "reason": "RL: Hit Stop Loss"}
        if state["hold_steps"] >= 50: return {"signal": "SELL_ALL", "reason": "RL: Timeout"}
        state["hold_steps"] += 1
        save_rl_state(state)

    if action == 1 and not state["in_position"]:
        dist = max(cur_atr * 2.0, price * 0.01)
        state.update({"in_position": True, "position_type": 1, "entry_price": price, "tp_price": price + dist, "sl_price": price - dist, "hold_steps": 0})
        save_rl_state(state)
        return {"signal": "BUY_ALL", "reason": "RL: Model Entry Signal"}
    elif action == 0 and state["in_position"]:
        state.update({"in_position": False, "qty": 0.0})
        save_rl_state(state)
        return {"signal": "SELL_ALL", "reason": "RL: Model Early Exit"}
        
    return {"signal": "HOLD", "reason": "RL: Model Wait"}

def compute_rl_features_v2(df: pd.DataFrame) -> np.ndarray:
    """Implement 18 features for Artemis V2 (Old version)"""
    prices = df['close'].values.astype(np.float32)
    highs, lows = df['high'].values.astype(np.float32), df['low'].values.astype(np.float32)
    volumes = df['volume'].values.astype(np.float32)
    log_ret = np.log(prices[1:] / (prices[:-1] + 1e-9))
    log_ret = np.concatenate([[0.0], log_ret])
    
    def get_rsi(p):
        delta = np.diff(p, prepend=p[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        ag = pd.Series(gain).ewm(span=14, adjust=False).mean().values
        al = pd.Series(loss).ewm(span=14, adjust=False).mean().values
        rs = ag / (al + 1e-9)
        return (100 - (100 / (1 + rs)) - 50) / 50

    def get_macd(p):
        s = pd.Series(p)
        f, sl = s.ewm(span=12, adjust=False).mean().values, s.ewm(span=26, adjust=False).mean().values
        m = f - sl
        sig = pd.Series(m).ewm(span=9, adjust=False).mean().values
        return m/(p+1e-9), sig/(p+1e-9), (m-sig)/(p+1e-9)

    def get_b(p):
        s = pd.Series(p)
        mid = s.rolling(20, min_periods=1).mean().values
        std = s.rolling(20, min_periods=1).std(ddof=0).fillna(0).values
        b = (p - (mid - 2*std)) / (4*std + 1e-9)
        return np.clip(b * 2 - 1, -1, 1)

    def get_atr(h, l, c):
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        return pd.Series(tr).ewm(span=14, adjust=False).mean().values / (c + 1e-9)

    rsi, (m, ms, mh), bb, atr = get_rsi(prices), get_macd(prices), get_b(prices), get_atr(highs, lows, prices)
    vol20 = pd.Series(log_ret).rolling(20, min_periods=1).std(ddof=0).values.astype(np.float32)
    mom5 = np.concatenate([[0.0]*5, (prices[5:] - prices[:-5]) / (prices[:-5] + 1e-9)])
    mom10 = np.concatenate([[0.0]*10, (prices[10:] - prices[:-10]) / (prices[:-10] + 1e-9)])
    v_avg = pd.Series(volumes).rolling(20, min_periods=1).mean()
    v_std = pd.Series(volumes).rolling(20, min_periods=1).std(ddof=0) + 1e-9
    vz = ((volumes - v_avg) / v_std).values.astype(np.float32)

    obs_batch = []
    for t in range(len(df)):
        obs_batch.append([log_ret[t], log_ret[max(t-1, 0)], log_ret[max(t-2, 0)], rsi[t], m[t], ms[t], mh[t], bb[t], atr[t], 0.0, 0.0, np.clip(mom5[t], -0.2, 0.2)/0.2, np.clip(mom10[t], -0.3, 0.3)/0.3, np.clip(vz[t], -3, 3)/3, np.clip(vol20[t]*100, 0, 5)/5, 0.0, 0.0, 0.0])
    return np.array(obs_batch, dtype=np.float32)

def run_rl_backtest(df: pd.DataFrame, model_path: str = "models/artemis_v2.zip") -> dict:
    """Artemis V2 Backtest (Restored implementation)"""
    if not RL_AVAILABLE or not os.path.exists(model_path):
        return {"strategy_name": "Artemis V2 (RL)", "error": "Model not found"}
    
    try:
        model = PPO.load(model_path)
        obs_matrix = compute_rl_features_v2(df)
        cash, holdings, equity_curve, trades_count, current_pos = float(INITIAL_CAPITAL), 0.0, [], 0, 0.0
        
        for i in range(len(df)):
            price = df.iloc[i]["close"]
            action, _ = model.predict(obs_matrix[i], deterministic=True)
            target_pos = float(np.clip(action[0], -1, 1)) * float(np.clip(action[1], 0, 1))
            
            if abs(target_pos - current_pos) > 0.1:
                target_usd = target_pos * (cash + holdings * price)
                diff_usd = target_usd - (holdings * price)
                if diff_usd > 5 and cash > diff_usd:
                    qty = (diff_usd - (diff_usd * FEE_RATE)) / price
                    holdings, cash, trades_count = holdings + qty, cash - diff_usd, trades_count + 1
                elif diff_usd < -5 and holdings > 0:
                    sell_qty = min(abs(diff_usd) / price, holdings)
                    cash, holdings, trades_count = cash + sell_qty * price * (1 - FEE_RATE), holdings - sell_qty, trades_count + 1
                current_pos = target_pos
            equity_curve.append(cash + holdings * price)
            
        final = cash + holdings * df.iloc[-1]["close"]
        peak, max_dd = INITIAL_CAPITAL, 0.0
        for v in equity_curve:
            if v > peak: peak = v
            dd = (peak - v) / peak
            if dd > max_dd: max_dd = dd
        return {"strategy_name": "Artemis V2 (RL)", "total_return_pct": round((final/INITIAL_CAPITAL-1)*100, 2), "max_drawdown_pct": round(max_dd*100, 2), "total_trades": trades_count}
    except Exception as e:
        return {"strategy_name": "Artemis V2 (RL)", "error": str(e)}

def get_live_signal(product_id: str):
    """Get live trading signal"""
    df     = fetch_coinbase_candles(product_id, days=200)
    df     = add_indicators(df).dropna().reset_index(drop=True)
    latest = df.iloc[-1]

    price  = latest["close"]
    ema_entry_s = latest["ema_entry_s"]
    ema_entry_l = latest["ema_entry_l"]
    ema_exit_s  = latest["ema_exit_s"]
    ema_exit_l  = latest["ema_exit_l"]
    rsi    = latest["rsi"]
    hist   = latest["macd_hist"]
    both_entry_up = latest["ema_entry_s_slope"] > 0 and latest["ema_entry_l_slope"] > 0

    signal = "HOLD"
    reason = []

    if latest["entry_golden_cross"] and rsi < 70 and hist > 0:
        signal = "BUY_ALL"
        reason.append(f"Entry: EMA{ENTRY_SHORT} crossed above EMA{ENTRY_LONG}")
    elif latest["exit_death_cross"]:
        signal = "SELL_ALL"
        reason.append(f"Exit: EMA{EXIT_SHORT} crossed below EMA{EXIT_LONG}")
    elif price < ema_exit_l * (1 - FULL_SELL_THRESHOLD):
        signal = "SELL_ALL"
        reason.append(f"Stop Loss: Price dropped below {EXIT_LONG}-day EMA by {FULL_SELL_THRESHOLD*100:.0f}%")
    elif price < ema_exit_s * (1 - HALF_SELL_THRESHOLD):
        signal = "SELL_HALF"
        reason.append(f"Partial Stop Loss: Price dropped below {EXIT_SHORT}-day EMA by {HALF_SELL_THRESHOLD*100:.0f}%")
    elif both_entry_up and (price >= ema_entry_s) and rsi < 70 and hist > 0:
        signal = "REBUY"
        reason.append("Regained entry EMA (Re-buy)")

    return {
        "timestamp": str(latest["date"]),
        "product": product_id,
        "price": round(price, 2),
        "ema_short": round(ema_entry_s, 2),
        "ema_long": round(ema_entry_l, 2),
        "rsi": round(rsi, 2),
        "macd_hist": round(hist, 4),
        "both_ema_up": bool(both_entry_up),
        "signal": signal,
        "reason": ", ".join(reason) if reason else "No trigger"
    }

# ─────────────────────────────────────────
# 6. Auxiliary: Virtual Account & Logs
# ─────────────────────────────────────────
def load_virtual_state():
    """Load virtual account state (USD & BTC)"""
    if os.path.exists(VIRTUAL_STATE_FILE):
        with open(VIRTUAL_STATE_FILE, "r") as f:
            return json.load(f)
    return {"usd": float(INITIAL_CAPITAL), "btc": 0.0}

def save_virtual_state(usd, btc):
    """Save virtual account state"""
    with open(VIRTUAL_STATE_FILE, "w") as f:
        json.dump({"usd": round(usd, 2), "btc": round(btc, 6)}, f)

def append_to_log(action, price, qty, balance_usd, reason):
    """Write trade record to CSV log"""
    file_exists = os.path.exists(TRADE_LOG_FILE)
    with open(TRADE_LOG_FILE, "a", encoding="utf-8") as f:
        # If new file, write header
        if not file_exists:
            f.write("timestamp,action,price,quantity,balance_usd,reason\n")
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{now},{action},{price},{qty},{balance_usd},{reason}\n"
        f.write(line)

def send_status_email(subject, body):
    """Send status notification Email"""
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    receivers = os.getenv("EMAIL_RECEIVER", "").split(",")
    server_addr = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", 587))

    if not all([sender, password, receivers]):
        print("Email settings incomplete, skipping email.")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = f"Coinbase Bot <{sender}>"
        msg['To'] = ", ".join(receivers)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Use SSL context to bypass macOS certificate issues
        import ssl
        context = ssl._create_unverified_context()
        
        with smtplib.SMTP(server_addr, port) as server:
            server.starttls(context=context)
            server.login(sender, password)
            server.sendmail(sender, receivers, msg.as_string())
        print(f"Notification email sent: {subject}")
    except Exception as e:
        print(f"Email failed: {e}")

# ─────────────────────────────────────────
# 7. Coinbase API Execution
# ─────────────────────────────────────────
def get_client():
    if not COINBASE_API_KEY or not COINBASE_API_SECRET:
        raise ValueError("Please set COINBASE_API_KEY and COINBASE_API_SECRET in .env")
    return RESTClient(api_key=COINBASE_API_KEY, api_secret=COINBASE_API_SECRET)

def get_balance(client, currency: str = "USD"):
    """Get available balance"""
    try:
        accounts = client.get_accounts()
        for acct in accounts["accounts"]:
            if acct["currency"] == currency:
                return float(acct["available_balance"]["value"])
    except Exception as e:
        print(f"Balance check failed: {e}")
    return 0.0

def execute_market_buy(client, product_id: str, amount_usd: float):
    """Execute Market Buy"""
    if DRY_RUN:
        print(f"[DRY RUN] Simulating Buy {product_id} Amount: ${amount_usd}")
        return {"order_id": "dry_run_buy"}
    
    try:
        # quote_size is the amount in USD
        order = client.market_order_buy(
            client_order_id=f"buy_{int(time.time())}",
            product_id=product_id,
            quote_size=str(round(amount_usd, 2))
        )
        print(f"Buy order sent: {order}")
        return order
    except Exception as e:
        print(f"Buy failed: {e}")
        return None

def execute_market_sell(client, product_id: str, amount_btc: float):
    """Execute Market Sell"""
    if DRY_RUN:
        print(f"[DRY RUN] Simulating Sell {product_id} Quantity: {amount_btc}")
        return {"order_id": "dry_run_sell"}

    try:
        # base_size is the quantity in BTC
        order = client.market_order_sell(
            client_order_id=f"sell_{int(time.time())}",
            product_id=product_id,
            base_size=str(amount_btc)
        )
        print(f"Sell order sent: {order}")
        return order
    except Exception as e:
        print(f"Sell failed: {e}")
        return None

# ─────────────────────────────────────────
# 7. Main Loop: Auto Trading
# ─────────────────────────────────────────
def run_auto_trading(product_id: str = None, interval_seconds: int = 3600):
    """
    Main Trading Loop
    product_id: Trading pair (e.g., BTC-USDC)
    interval_seconds: Signal check interval (Default 1 hour)
    """
    if product_id is None:
        product_id = PRODUCT_ID
        
    print(f"\n[START] Auto Trading | Pair: {product_id} | Mode: {'DRY RUN' if DRY_RUN else 'REAL'}")
    
    client = get_client()
    base_currency = product_id.split("-")[0]  # e.g., BTC
    quote_currency = product_id.split("-")[1] # e.g., USD

    # Initialize balances
    if DRY_RUN:
        v_state = load_virtual_state()
        usd_bal = v_state["usd"]
        btc_bal = v_state["btc"]
        print(f"Loaded virtual balances: ${usd_bal} USD, {btc_bal} {base_currency}")
    else:
        usd_bal = get_balance(client, quote_currency)
        btc_bal = get_balance(client, base_currency)

    # Track partial sell state
    state = {"half_sold": False}

    while True:
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{now}] Checking signals...")
            
            sig_data = get_live_signal(product_id)
            signal = sig_data["signal"]
            price = sig_data["price"]
            print(f"Price: ${price} | Signal: {signal} ({sig_data['reason']})")

            # Send notification email after each check
            email_subject = f"[Coinbase] {product_id}: {signal} (${price})"
            email_body = f"Time: {now}\nPrice: ${price}\nSignal: {signal}\nReason: {sig_data['reason']}\nMode: {'DRY RUN' if DRY_RUN else 'REAL'}"
            send_status_email(email_subject, email_body)

            # Update current real/virtual balances
            if not DRY_RUN:
                usd_bal = get_balance(client, quote_currency)
                btc_bal = get_balance(client, base_currency)

            if signal == "BUY_ALL" or signal == "REBUY":
                if usd_bal > 10:  # Min buy amount
                    print(f"Buy signal detected, balance: ${usd_bal}")
                    order = execute_market_buy(client, product_id, usd_bal)
                    if order:
                        qty_bought = (usd_bal * (1 - FEE_RATE)) / price
                        if DRY_RUN:
                            btc_bal += qty_bought
                            usd_bal = 0.0
                            save_virtual_state(usd_bal, btc_bal)
                        
                        append_to_log("BUY", price, round(qty_bought, 6), round(usd_bal, 2), sig_data["reason"])
                        state["half_sold"] = False
                else:
                    print(f"Insufficient funds (${usd_bal}), skipping buy")

            elif signal == "SELL_ALL":
                if btc_bal > 0.0001:  # Min sell amount
                    print(f"Sell signal detected, holdings: {btc_bal} {base_currency}")
                    order = execute_market_sell(client, product_id, btc_bal)
                    if order:
                        revenue = btc_bal * price * (1 - FEE_RATE)
                        if DRY_RUN:
                            usd_bal += revenue
                            btc_bal = 0.0
                            save_virtual_state(usd_bal, btc_bal)
                        
                        append_to_log("SELL_ALL", price, round(btc_bal, 6), round(usd_bal, 2), sig_data["reason"])
                        state["half_sold"] = False
                else:
                    print(f"No holdings, nothing to sell")

            elif signal == "SELL_HALF":
                if not state["half_sold"]:
                    if btc_bal > 0.0001:
                        sell_qty = btc_bal / 2
                        print(f"Partial sell signal, selling half: {sell_qty}")
                        order = execute_market_sell(client, product_id, sell_qty)
                        if order:
                            revenue = sell_qty * price * (1 - FEE_RATE)
                            if DRY_RUN:
                                usd_bal += revenue
                                btc_bal -= sell_qty
                                save_virtual_state(usd_bal, btc_bal)
                            
                            append_to_log("SELL_HALF", price, round(sell_qty, 6), round(usd_bal, 2), sig_data["reason"])
                            state["half_sold"] = True
                else:
                    print("Partial sell already executed, skipping")

            else:
                print("HOLD")

        except Exception as e:
            print(f"Error in trading loop: {e}")

        print(f"Waiting {interval_seconds // 60} minutes for next check...")
        time.sleep(interval_seconds)

# ─────────────────────────────────────────
# 8. Execution & Reporting
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Support command line args (e.g.: python coinbase_strategy.py 1)
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("=" * 55)
        print("  Coinbase Automated Trading System (EMA+RSI+MACD)")
        print("=" * 55)
        print("  1. Run Historical Backtest")
        print("  2. Start Auto Trading (MA Strategy)")
        print("  3. Start Auto Trading (Artemis RL Strategy)")
        print("=" * 55)
        try:
            choice = input("Choice (1/2/3): ").strip()
        except EOFError:
            choice = ""

    if choice == "1":
        print("\n--- Backtest Settings ---")
        if len(sys.argv) > 2:
            time_input = sys.argv[2]
        else:
            time_input = input("Enter days (e.g., 300) or start date (YYYY-MM-DD) [Default 300]: ").strip()
        
        if not time_input:
            time_input = "300"
            
        print("[Backtest] Fetching data and calculating...")
        
        # Check if days or date
        if "-" in time_input:
            # Add 60 days buffer for indicator warmup
            try:
                dt_requested = datetime.strptime(time_input, "%Y-%m-%d")
                buffer_start = (dt_requested - timedelta(days=60)).strftime("%Y-%m-%d")
                print(f"[System] Fetching buffer data starting from {buffer_start} for warmup...")
                df_raw = fetch_coinbase_candles("BTC-USD", start_date=buffer_start)
            except Exception:
                df_raw = fetch_coinbase_candles("BTC-USD", start_date=time_input)
        else:
            try:
                days = int(time_input)
                # Also add 60 days buffer
                df_raw = fetch_coinbase_candles("BTC-USD", days=days + 60)
            except ValueError:
                df_raw = fetch_coinbase_candles("BTC-USD", days=300 + 60)
            
        # Run all strategies
        df_ind = add_indicators(df_raw).dropna().reset_index(drop=True)
        
        # Slice buffer, ensure statistics start from requested date
        if "-" in time_input:
            try:
                target_dt = pd.to_datetime(time_input).tz_localize(None)
                df_ind['date_no_tz'] = pd.to_datetime(df_ind['date']).dt.tz_localize(None)
                df_ind = df_ind[df_ind['date_no_tz'] >= target_dt].copy()
                df_ind = df_ind.drop(columns=['date_no_tz']).reset_index(drop=True)
            except Exception as e:
                print(f"[Warning] Date slicing failed: {e}")
        
        if df_ind.empty:
            print("[Error] Not enough data for this period.")
            import sys
            sys.exit(1)

        res_dual = run_backtest(df_ind)
        res_ma   = run_ma_cross_backtest(df_ind)
        res_grid = run_grid_backtest(df_ind)
        res_rl_v2 = run_rl_backtest(df_ind) # Original V2
        res_rl_tb = run_rl_backtest_tb(df_ind) # New Triple-Barrier
        
        print("\n" + "="*60)
        print(f"Strategy Performance Comparison ({res_dual['start_date']} to {res_dual['end_date']})")
        print("="*60)
        print(f"{'Strategy Name':<25} | {'Return':<8} | {'Max DD':<8} | {'Trades':<4}")
        print("-" * 60)
        
        def print_row(r):
            name = r.get("strategy_name", "Unknown")
            if "error" in r:
                print(f"{name:<25} | {'ERROR':<8} | {'---':<8} | {'---':<4}")
                return
            ret  = f"{r.get('total_return_pct', 0.0):+.2f}%"
            dd   = f"{r.get('max_drawdown_pct', 0.0):.2f}%"
            cnt  = r.get("total_trades", 0)
            print(f"{name:<25} | {ret:<8} | {dd:<8} | {cnt:<4}")

        print_row(res_dual)
        print_row(res_ma)
        print_row(res_grid)
        print_row(res_rl_v2)
        print_row(res_rl_tb)
        
        # B&H Reference
        bh_ret = f"{res_dual['buy_hold_return_pct']:+.2f}%"
        print(f"{'Buy & Hold (Ref)':<25} | {bh_ret:<8} | {'N/A':<8} | {'0':<4}")
        
        print("-" * 60)
        print(f"Final Equity Preview (Dual EMA): ${res_dual['final_equity']} USD")
        print("="*60)
        print("[Done] Backtest complete.")

    elif choice == "2":
        run_auto_trading(PRODUCT_ID, interval_seconds=3600)

    elif choice == "3":
        # Start Artemis Triple-Barrier RL Auto Trading
        def run_rl_auto_loop(product_id=PRODUCT_ID):
            print(f"\n[START] Artemis RL Auto Trading | Mode: {'DRY RUN' if DRY_RUN else 'REAL'}")
            client = get_client()
            while True:
                try:
                    sig_data = get_rl_signal_tb(product_id)
                    signal, reason = sig_data["signal"], sig_data["reason"]
                    print(f"[{datetime.now()}] Signal: {signal} ({reason})")
                    
                    if signal == "BUY_ALL":
                        bal = get_balance(client, product_id.split("-")[1])
                        execute_market_buy(client, product_id, bal)
                    elif signal == "SELL_ALL":
                        hold = get_balance(client, product_id.split("-")[0])
                        execute_market_sell(client, product_id, hold)
                        
                except Exception as e: print(f"Error: {e}")
                time.sleep(3600)
        run_rl_auto_loop()
        
    else:
        print("Exit.")

