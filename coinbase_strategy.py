"""
Coinbase 自動交易策略 - EMA 黃金交叉 + RSI + MACD
=====================================================
參數設定：
  短線 EMA : 20日
  長線 EMA : 50日
  RSI      : 14期
  MACD     : 12 / 26 / 9
  Coinbase 手續費 : 0.6% (Taker fee，進出各一次)

策略規則：
  買入條件：
    1. EMA20 向上穿越 EMA50（黃金交叉）
    2. RSI < 70（非超買）
    3. MACD histogram > 0（動能向上）

  賣出條件（優先序）：
    A. EMA20 向下穿越 EMA50（死亡交叉）→ 全部賣出
    B. 價格跌破 EMA50 × (1 - 3%)   → 全部賣出
    C. 價格跌破 EMA20 × (1 - 3%)   → 賣出 50%

  再買回條件（部分倉位因C被賣出後）：
    - EMA20 & EMA50 仍向上（slope > 0）
    - 價格重新站回 EMA20 以上 或 觸及 EMA50 出現反彈
    - RSI < 70, MACD histogram > 0
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
import json
import os
import time
from dotenv import load_dotenv
import sys
from coinbase.rest import RESTClient

# 載入環境變數
load_dotenv()

# RL 相關匯入
try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")
DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"

# ─────────────────────────────────────────
# 1. 參數 (進場 20/50, 退場 5/10)
# ─────────────────────────────────────────
ENTRY_SHORT = 20
ENTRY_LONG  = 50
EXIT_SHORT  = 5
EXIT_LONG   = 10

RSI_PERIOD  = 14
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9
FEE_RATE    = 0.006   # Coinbase Taker fee 0.6% 單邊；買+賣 = 1.2%
HALF_SELL_THRESHOLD  = 0.03   # 跌破短線 3%
FULL_SELL_THRESHOLD  = 0.03   # 跌破長線 3%
INITIAL_CAPITAL = 500      # 初始資金 (USD)

VIRTUAL_STATE_FILE = "virtual_state.json"
TRADE_LOG_FILE     = "trade_log.csv"

# ─────────────────────────────────────────
# 2. 取得歷史資料（Coinbase Advanced API）
# ─────────────────────────────────────────
def fetch_coinbase_candles(product_id: str = "BTC-USD",
                            granularity: str = "ONE_DAY",
                            days: int = 300,
                            start_date: str = None) -> pd.DataFrame:
    """
    從 Coinbase Advanced Trade API 取得 K 線資料
    支援分段抓取長時段資料
    """
    now_ts = int(datetime.utcnow().timestamp())
    
    if start_date:
        try:
            dt = datetime.strptime(start_date, "%Y-%m-%d")
            requested_start_ts = int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            print(f"日期格式錯誤 ({start_date})，改用天數設定: {days}")
            requested_start_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    else:
        requested_start_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp())

    all_data = []
    current_end = now_ts
    
    # Coinbase API 限制單次約 300 根 K 線 (對 ONE_DAY 來說約 300 天)
    # 我們分段抓取，每次抓 300 天，直到覆蓋到 requested_start_ts
    chunk_seconds = 300 * 24 * 3600 # 300 天
    
    while current_end > requested_start_ts:
        current_start = max(requested_start_ts, current_end - chunk_seconds)
        
        url = (f"https://api.coinbase.com/api/v3/brokerage/market/products/"
               f"{product_id}/candles"
               f"?start={current_start}&end={current_end}&granularity={granularity}")

        headers = {"Content-Type": "application/json"}
        resp = requests.get(url, headers=headers, timeout=60)
        
        if resp.status_code != 200:
            print(f"抓取資料失敗 ({current_start} to {current_end}): {resp.text}")
            break
            
        data = resp.json().get("candles", [])
        if not data:
            break
            
        all_data.extend(data)
        # 更新下一次的結束時間為本次的最早時間 - 1
        current_end = current_start - 1
        
        # 避免太頻繁呼叫
        time.sleep(0.5)

    if not all_data:
        raise ValueError("無法取得任何歷史資料，請檢查網路或日期設定。")

    df = pd.DataFrame(all_data, columns=["start", "low", "high", "open", "close", "volume"])
    df["date"]  = pd.to_datetime(df["start"].astype(int), unit="s")
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date","open","high","low","close","volume"]]

# ─────────────────────────────────────────
# 3. 技術指標計算
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
    # 進場指標
    df["ema_entry_s"] = calc_ema(df["close"], ENTRY_SHORT)
    df["ema_entry_l"] = calc_ema(df["close"], ENTRY_LONG)
    # 退場指標
    df["ema_exit_s"]  = calc_ema(df["close"], EXIT_SHORT)
    df["ema_exit_l"]  = calc_ema(df["close"], EXIT_LONG)
    
    # 額外指標 (用於其他策略比較)
    df["ma5"]  = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    
    df["rsi"] = calc_rsi(df["close"], RSI_PERIOD)
    df["macd"], df["macd_signal"], df["macd_hist"] = calc_macd(
        df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    # 進場交叉 (Dual EMA)
    df["entry_golden_cross"] = (df["ema_entry_s"] > df["ema_entry_l"]) & \
                               (df["ema_entry_s"].shift(1) <= df["ema_entry_l"].shift(1))
    
    # 退場交叉 (Dual EMA)
    df["exit_death_cross"]   = (df["ema_exit_s"] < df["ema_exit_l"]) & \
                               (df["ema_exit_s"].shift(1) >= df["ema_exit_l"].shift(1))
    
    # 斜率用於再買回判斷
    df["ema_entry_s_slope"] = df["ema_entry_s"].diff()
    df["ema_entry_l_slope"] = df["ema_entry_l"].diff()
    
    return df

# ─────────────────────────────────────────
# 4. 回測引擎
# ─────────────────────────────────────────
def run_backtest(df: pd.DataFrame) -> dict:
    df = add_indicators(df).dropna().reset_index(drop=True)

    cash        = float(INITIAL_CAPITAL)
    holdings    = 0.0        # 持有數量（BTC）
    avg_cost    = 0.0        # 平均成本
    half_sold   = False      # 是否已執行半賣
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

        # ── 賣出邏輯 (使用 5/10 退場訊號) ──────────────────────
        if holdings > 1e-8:
            # A: 5/10 死亡交叉 → 全賣
            if row["exit_death_cross"]:
                sell(i, price, "5/10 死亡交叉 (快退)", frac=1.0)
                half_sold = False
                continue

            # B: 跌破 10 日線 3% → 全賣 (硬止損)
            if price < ema_exit_l * (1 - FULL_SELL_THRESHOLD):
                sell(i, price, f"跌破 10 日線-{int(FULL_SELL_THRESHOLD*100)}%", frac=1.0)
                half_sold = False
                continue

            # C: 跌破 5 日線 3% → 賣一半
            if price < ema_exit_s * (1 - HALF_SELL_THRESHOLD) and not half_sold:
                sell(i, price, f"跌破 5 日線-{int(HALF_SELL_THRESHOLD*100)}%", frac=0.5)

        # ── 買入邏輯 (使用 20/50 進場訊號) ────────────────────────────
        buy_signal = (row["entry_golden_cross"] and rsi < 70 and hist > 0)

        if buy_signal and holdings < 1e-8:
            buy(i, price, "20/50 黃金交叉 (強進)", frac=1.0)

        # ── 再買回（半倉賣出後，反彈條件，參考進場均線）─────
        rebuy_signal = half_sold and both_entry_up and (price >= ema_entry_s) and rsi < 70 and hist > 0
        if rebuy_signal:
            buy(i, price, "站回 20 日線再買回", frac=1.0)

    # 最後收盤平倉
    if holdings > 1e-8:
        last = df.iloc[-1]
        sell(len(df)-1, last["close"], "回測結束平倉", frac=1.0)

    final_equity = cash
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    # 計算 Buy & Hold：從策略開始交易的那一點（LONG_EMA 之後）到最後
    # 這樣對比才公平，因為策略需要熱身時間
    start_price = df.iloc[0]["close"]
    end_price   = df.iloc[-1]["close"]
    buy_hold     = (end_price / start_price - 1) * 100

    # 最大回撤
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
    """
    實作 5MA/10MA 交叉策略 (來自 Sideproject)
    - 5MA > 10MA: 買入
    - 價格 < 5MA * (1-buffer): 賣出 50%
    - 價格 < 10MA * (1-buffer): 賣出 100%
    """
    df = df.dropna().reset_index(drop=True)
    cash = float(INITIAL_CAPITAL)
    holdings = 0.0
    trades = []
    equity_curve = []

    for i in range(len(df)):
        row = df.loc[i]
        price = row["close"]
        m5 = row["ma5"]
        m10 = row["ma10"]

        # Buy Signal
        if m5 > m10 and cash > 1:
            fee = cash * FEE_RATE
            qty = (cash - fee) / price
            holdings += qty
            cash = 0
            trades.append({"action": "BUY", "price": price})

        # Sell Signal
        elif holdings > 1e-8 and price < m5 * (1 - buffer_pct):
            if price < m10 * (1 - buffer_pct):
                # 全賣
                rev = holdings * price * (1 - FEE_RATE)
                cash += rev
                holdings = 0
                trades.append({"action": "SELL_ALL", "price": price})
            else:
                # 賣一半
                qty_half = holdings * 0.5
                rev = qty_half * price * (1 - FEE_RATE)
                cash += rev
                holdings -= qty_half
                trades.append({"action": "SELL_HALF", "price": price})

        equity = cash + holdings * price
        equity_curve.append(equity)

    final_equity = cash + holdings * df.iloc[-1]["close"]
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    # 計算最大回撤
    peak, max_dd = INITIAL_CAPITAL, 0.0
    for v in equity_curve:
        if v > peak: peak = v
        dd = (peak - v) / peak
        if dd > max_dd: max_dd = dd

    return {
        "strategy_name": "5/10 MA Cross (User)",
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_trades": len(trades)
    }

def run_grid_backtest(df: pd.DataFrame, grid_count: int = 10, width_pct: float = 0.05) -> dict:
    """
    實作靜態網格交易策略 (來自 Sideproject)
    """
    df = df.dropna().reset_index(drop=True)
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
        
        # 模擬網格觸發 (簡化版)
        for lv in levels:
            # 價格下穿過網格線 -> 買入 (假設每格投入 10% 資金)
            if current_price > lv and low <= lv:
                buy_amt = INITIAL_CAPITAL / grid_count
                if cash >= buy_amt:
                    qty = (buy_amt * (1 - FEE_RATE)) / lv
                    holdings += qty
                    cash -= buy_amt
                    trades_count += 1
            
            # 價格上穿過網格線 -> 賣出
            if current_price < lv and high >= lv:
                if holdings > 0:
                    sell_qty = holdings / grid_count # 簡化：賣出總持倉的 1/N
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

# ─────────────────────────────────────────
# 5. 強化學習策略 (Artemis V2 RL)
# ─────────────────────────────────────────

def compute_rl_features(df: pd.DataFrame) -> np.ndarray:
    """
    實作 Artemis V2 的 18 種特徵計算
    """
    prices = df['close'].values.astype(np.float32)
    highs = df['high'].values.astype(np.float32)
    lows = df['low'].values.astype(np.float32)
    volumes = df['volume'].values.astype(np.float32)
    
    # 1. Log Returns
    log_ret = np.log(prices[1:] / (prices[:-1] + 1e-9))
    log_ret = np.concatenate([[0.0], log_ret])
    
    # 2. RSI (Normalized to [-1, 1])
    def get_rsi(p, period=14):
        delta = np.diff(p, prepend=p[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
        rs = np.where(avg_loss == 0, 100.0, avg_gain / (avg_loss + 1e-9))
        rsi = 100 - (100 / (1 + rs))
        return (rsi - 50) / 50

    # 3. MACD
    def get_macd(p):
        s = pd.Series(p)
        f = s.ewm(span=12, adjust=False).mean().values
        sl = s.ewm(span=26, adjust=False).mean().values
        m = f - sl
        sig = pd.Series(m).ewm(span=9, adjust=False).mean().values
        h = m - sig
        return m / (p + 1e-9), sig / (p + 1e-9), h / (p + 1e-9)

    # 4. Bollinger
    def get_b(p):
        s = pd.Series(p)
        mid = s.rolling(20, min_periods=1).mean().values
        std = s.rolling(20, min_periods=1).std(ddof=0).fillna(0).values
        u, l = mid + 2*std, mid - 2*std
        b = (p - l) / (u - l + 1e-9)
        return np.clip(b * 2 - 1, -1, 1)

    # 5. ATR
    def get_atr(h, l, c):
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        a = pd.Series(tr).ewm(span=14, adjust=False).mean().values
        return a / (c + 1e-9)

    rsi = get_rsi(prices)
    m, ms, mh = get_macd(prices)
    bb = get_b(prices)
    atr = get_atr(highs, lows, prices)
    vol20 = pd.Series(log_ret).rolling(20, min_periods=1).std(ddof=0).values.astype(np.float32)
    mom5 = np.concatenate([[0.0]*5, (prices[5:] - prices[:-5]) / (prices[:-5] + 1e-9)])
    mom10 = np.concatenate([[0.0]*10, (prices[10:] - prices[:-10]) / (prices[:-10] + 1e-9)])
    v_avg = pd.Series(volumes).rolling(20, min_periods=1).mean()
    v_std = pd.Series(volumes).rolling(20, min_periods=1).std(ddof=0) + 1e-9
    vz = ((volumes - v_avg) / v_std).values.astype(np.float32)

    # 組合 18 維特徵 (簡化版，省略部分動態狀態以適配靜態回測)
    obs_batch = []
    for t in range(len(df)):
        obs = [
            log_ret[t],
            log_ret[max(t-1, 0)],
            log_ret[max(t-2, 0)],
            rsi[t],
            m[t],
            ms[t],
            mh[t],
            bb[t],
            atr[t],
            0.0, # Current Position (Mock)
            0.0, # Unrealized PnL (Mock)
            np.clip(mom5[t], -0.2, 0.2) / 0.2,
            np.clip(mom10[t], -0.3, 0.3) / 0.3,
            np.clip(vz[t], -3, 3) / 3,
            np.clip(vol20[t] * 100, 0, 5) / 5,
            0.0, # Steps in position
            0.0, # Portfolio growth
            0.0  # Max drawdown
        ]
        obs_batch.append(obs)
    return np.array(obs_batch, dtype=np.float32)

def run_rl_backtest(df: pd.DataFrame, model_path: str = "models/artemis_v2.zip") -> dict:
    """
    執行 Artemis V2 RL 模型的模擬回測
    """
    if not RL_AVAILABLE or not os.path.exists(model_path):
        return {"strategy_name": "Artemis V2 (RL)", "total_return_pct": 0.0, "max_drawdown_pct": 0.0, "total_trades": 0, "error": "Model or SB3 not found"}

    try:
        model = PPO.load(model_path)
    except Exception as e:
        return {"strategy_name": "Artemis V2 (RL)", "total_return_pct": 0.0, "max_drawdown_pct": 0.0, "total_trades": 0, "error": str(e)}

    obs_matrix = compute_rl_features(df)
    cash = float(INITIAL_CAPITAL)
    holdings = 0.0
    equity_curve = []
    trades_count = 0
    current_pos_size = 0.0 # -1 to 1

    for i in range(len(df)):
        price = df.iloc[i]["close"]
        obs = obs_matrix[i]
        
        # 預測動作
        action, _ = model.predict(obs, deterministic=True)
        direction = float(np.clip(action[0], -1, 1))
        size_frac  = float(np.clip(action[1], 0, 1))
        target_pos_size = direction * size_frac
        
        # 執行交易 (簡化：目標倉位法)
        if abs(target_pos_size - current_pos_size) > 0.1:
            # 計算買賣量
            target_usd = target_pos_size * (cash + holdings * price)
            diff_usd = target_usd - (holdings * price)
            
            if diff_usd > 5 and cash > diff_usd: # BUY
                fee = diff_usd * FEE_RATE
                qty = (diff_usd - fee) / price
                holdings += qty
                cash -= diff_usd
                trades_count += 1
            elif diff_usd < -5 and holdings > 0: # SELL
                sell_qty = abs(diff_usd) / price
                if sell_qty > holdings: sell_qty = holdings
                cash += sell_qty * price * (1 - FEE_RATE)
                holdings -= sell_qty
                trades_count += 1
            
            current_pos_size = target_pos_size

        equity = cash + holdings * price
        equity_curve.append(equity)

    final_equity = cash + holdings * df.iloc[-1]["close"]
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    max_dd = 0.0
    peak = INITIAL_CAPITAL
    for v in equity_curve:
        if v > peak: peak = v
        dd = (peak - v) / peak
        if dd > max_dd: max_dd = dd

    return {
        "strategy_name": "Artemis V2 (RL)",
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_trades": trades_count
    }
    """取得即時交易訊號"""
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
        reason.append(f"進場訊號: EMA{ENTRY_SHORT} 向上打穿 EMA{ENTRY_LONG}")
    elif latest["exit_death_cross"]:
        signal = "SELL_ALL"
        reason.append(f"退場訊號: EMA{EXIT_SHORT} 向下打穿 EMA{EXIT_LONG}")
    elif price < ema_exit_l * (1 - FULL_SELL_THRESHOLD):
        signal = "SELL_ALL"
        reason.append(f"硬止損: 價格跌破 {EXIT_LONG} 日線 {FULL_SELL_THRESHOLD*100:.0f}%")
    elif price < ema_exit_s * (1 - HALF_SELL_THRESHOLD):
        signal = "SELL_HALF"
        reason.append(f"部分止損: 價格跌破 {EXIT_SHORT} 日線 {HALF_SELL_THRESHOLD*100:.0f}%")
    elif both_entry_up and (price >= ema_entry_s) and rsi < 70 and hist > 0:
        signal = "REBUY"
        reason.append("重新站回進場短均線再買回")

    return {
        "timestamp": str(latest["date"]),
        "product": product_id,
        "price": round(price, 2),
        "ema_short": round(ema_s, 2),
        "ema_long": round(ema_l, 2),
        "rsi": round(rsi, 2),
        "macd_hist": round(hist, 4),
        "both_ema_up": bool(both_up),
        "signal": signal,
        "reason": ", ".join(reason) if reason else "無觸發條件"
    }

# ─────────────────────────────────────────
# 6. 輔助功能：虛擬帳戶與日誌
# ─────────────────────────────────────────
def load_virtual_state():
    """載入虛擬帳戶狀態 (USD & BTC)"""
    if os.path.exists(VIRTUAL_STATE_FILE):
        with open(VIRTUAL_STATE_FILE, "r") as f:
            return json.load(f)
    return {"usd": float(INITIAL_CAPITAL), "btc": 0.0}

def save_virtual_state(usd, btc):
    """儲存虛擬帳戶狀態"""
    with open(VIRTUAL_STATE_FILE, "w") as f:
        json.dump({"usd": round(usd, 2), "btc": round(btc, 6)}, f)

def append_to_log(action, price, qty, balance_usd, reason):
    """將交易紀錄寫入 CSV 日誌"""
    file_exists = os.path.exists(TRADE_LOG_FILE)
    with open(TRADE_LOG_FILE, "a", encoding="utf-8") as f:
        # 如果是新檔案，寫入標題
        if not file_exists:
            f.write("timestamp,action,price,quantity,balance_usd,reason\n")
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{now},{action},{price},{qty},{balance_usd},{reason}\n"
        f.write(line)

# ─────────────────────────────────────────
# 7. Coinbase API 交易執行
# ─────────────────────────────────────────
def get_client():
    if not COINBASE_API_KEY or not COINBASE_API_SECRET:
        raise ValueError("請先在 .env 檔案中設定 COINBASE_API_KEY 與 COINBASE_API_SECRET")
    return RESTClient(api_key=COINBASE_API_KEY, api_secret=COINBASE_API_SECRET)

def get_balance(client, currency: str = "USD"):
    """取得帳戶可用餘額"""
    try:
        accounts = client.get_accounts()
        for acct in accounts["accounts"]:
            if acct["currency"] == currency:
                return float(acct["available_balance"]["value"])
    except Exception as e:
        print(f"查詢餘額失敗: {e}")
    return 0.0

def execute_market_buy(client, product_id: str, amount_usd: float):
    """執行市價買入"""
    if DRY_RUN:
        print(f"[DRY RUN] 模擬買入 {product_id} 金額: ${amount_usd}")
        return {"order_id": "dry_run_buy"}
    
    try:
        # quote_size 是以 USD 為單位的買入金額
        order = client.market_order_buy(
            client_order_id=f"buy_{int(time.time())}",
            product_id=product_id,
            quote_size=str(round(amount_usd, 2))
        )
        print(f"買入訂單已發送: {order}")
        return order
    except Exception as e:
        print(f"買入失敗: {e}")
        return None

def execute_market_sell(client, product_id: str, amount_btc: float):
    """執行市價賣出"""
    if DRY_RUN:
        print(f"[DRY RUN] 模擬賣出 {product_id} 數量: {amount_btc}")
        return {"order_id": "dry_run_sell"}

    try:
        # base_size 是以 BTC 為單位的賣出數量
        order = client.market_order_sell(
            client_order_id=f"sell_{int(time.time())}",
            product_id=product_id,
            base_size=str(amount_btc)
        )
        print(f"賣出訂單已發送: {order}")
        return order
    except Exception as e:
        print(f"賣出失敗: {e}")
        return None

# ─────────────────────────────────────────
# 7. 主程式：自動交易迴圈
# ─────────────────────────────────────────
def run_auto_trading(product_id: str = "BTC-USD", interval_seconds: int = 3600):
    """
    主交易迴圈
    product_id: 交易對 (例如 BTC-USD)
    interval_seconds: 檢查訊號的時間間隔 (預設 1 小時)
    """
    print(f"\n[啟動] 自動交易程式 | 交易對: {product_id} | 模式: {'模擬 (Dry Run)' if DRY_RUN else '真實'}")
    
    client = get_client()
    base_currency = product_id.split("-")[0]  # e.g., BTC
    quote_currency = product_id.split("-")[1] # e.g., USD

    # 初始化餘額
    if DRY_RUN:
        v_state = load_virtual_state()
        usd_bal = v_state["usd"]
        btc_bal = v_state["btc"]
        print(f"載入模擬餘額: ${usd_bal} USD, {btc_bal} {base_currency}")
    else:
        usd_bal = get_balance(client, quote_currency)
        btc_bal = get_balance(client, base_currency)

    # 用於追踪部分平倉狀態
    state = {"half_sold": False}

    while True:
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{now}] 正在檢查訊號...")
            
            sig_data = get_live_signal(product_id)
            signal = sig_data["signal"]
            price = sig_data["price"]
            print(f"目前價格: ${price} | 訊號: {signal} ({sig_data['reason']})")

            # 更新當前實際/模擬餘額
            if not DRY_RUN:
                usd_bal = get_balance(client, quote_currency)
                btc_bal = get_balance(client, base_currency)

            if signal == "BUY_ALL" or signal == "REBUY":
                if usd_bal > 10:  # 最小買入金額
                    print(f"偵測到買入訊號，餘額: ${usd_bal}")
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
                    print(f"現金不足 (${usd_bal})，無法執行買入")

            elif signal == "SELL_ALL":
                if btc_bal > 0.0001:  # 最小賣出量
                    print(f"偵測到賣出訊號，持倉: {btc_bal} {base_currency}")
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
                    print(f"無持倉，無需賣出")

            elif signal == "SELL_HALF":
                if not state["half_sold"]:
                    if btc_bal > 0.0001:
                        sell_qty = btc_bal / 2
                        print(f"偵測到部分賣出訊號，賣出一半持倉: {sell_qty}")
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
                    print("先前已執行過部分賣出，略過")

            else:
                print("保持觀望 (HOLD)")

        except Exception as e:
            print(f"執行交易迴圈時出錯: {e}")

        print(f"等待 {interval_seconds // 60} 分鐘後進行下一次檢查...")
        time.sleep(interval_seconds)

# ─────────────────────────────────────────
# 8. 執行回測與展示報告
# ─────────────────────────────────────────
if __name__ == "__main__":
    # 支援命令列參數 (例如: python coinbase_strategy.py 1)
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("=" * 55)
        print("  Coinbase 自動交易系統 (EMA+RSI+MACD)")
        print("=" * 55)
        print("  1. 執行歷史回測")
        print("  2. 啟動自動交易 (1 小時檢查一次)")
        print("  3. 啟動自動交易 (自訂頻率)")
        print("=" * 55)
        try:
            choice = input("請選擇功能 (1/2/3): ").strip()
        except EOFError:
            choice = ""

    if choice == "1":
        print("\n--- 回測設定 ---")
        if len(sys.argv) > 2:
            time_input = sys.argv[2]
        else:
            time_input = input("請輸入回測天數 (例如 300) 或 開始日期 (YYYY-MM-DD) [預設 300]: ").strip()
        
        if not time_input:
            time_input = "300"
            
        print("[回測] 正在獲取資料並計算...")
        
        # 判斷是天數還是日期
        if "-" in time_input:
            df_raw = fetch_coinbase_candles("BTC-USD", start_date=time_input)
        else:
            try:
                days = int(time_input)
            except ValueError:
                days = 300
            df_raw = fetch_coinbase_candles("BTC-USD", days=days)
            
        # 跑所有策略
        df_ind = add_indicators(df_raw)
        res_dual = run_backtest(df_ind)
        res_ma   = run_ma_cross_backtest(df_ind)
        res_grid = run_grid_backtest(df_ind)
        res_rl   = run_rl_backtest(df_ind)
        
        print("\n" + "="*60)
        print(f"策略效能對比報告 ({res_dual['start_date']} 至 {res_dual['end_date']})")
        print("="*60)
        print(f"{'策略名稱':<25} | {'報酬率':<8} | {'最大回撤':<8} | {'交易次數':<4}")
        print("-" * 60)
        
        def print_row(r):
            name = r["strategy_name"]
            if "error" in r:
                print(f"{name:<25} | {'ERROR':<8} | {'---':<8} | {'---':<4}")
                return
            ret  = f"{r['total_return_pct']:+.2f}%"
            dd   = f"{r['max_drawdown_pct']}%"
            cnt  = r["total_trades"]
            print(f"{name:<25} | {ret:<8} | {dd:<8} | {cnt:<4}")

        print_row(res_dual)
        print_row(res_ma)
        print_row(res_grid)
        print_row(res_rl)
        
        # B&H 參考
        bh_ret = f"{res_dual['buy_hold_return_pct']:+.2f}%"
        print(f"{'Buy & Hold (參考)':<25} | {bh_ret:<8} | {'N/A':<8} | {'0':<4}")
        
        print("-" * 60)
        print(f"最終資產預覽 (Dual EMA): ${res_dual['final_equity']} USD")
        print("="*60)
        print("[完成] 回測結束。")

    elif choice == "2":
        run_auto_trading("BTC-USD", interval_seconds=3600)

    elif choice == "3":
        if len(sys.argv) > 2:
            mins = sys.argv[2]
        else:
            mins = input("輸入檢查頻率 (分鐘, 預設 60): ").strip()
        
        try:
            mins = int(mins) if mins else 60
        except ValueError:
            mins = 60
            
        run_auto_trading("BTC-USD", interval_seconds=mins * 60)
        
    else:
        print("結束。")

