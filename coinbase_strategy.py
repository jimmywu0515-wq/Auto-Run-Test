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
    - 價格觸及 EMA50 並出現反彈（當日收盤 > EMA50）
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
from coinbase.rest import RESTClient

# 載入環境變數
load_dotenv()

COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")
DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"

# ─────────────────────────────────────────
# 1. 參數
# ─────────────────────────────────────────
SHORT_EMA   = 20
LONG_EMA    = 50
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
                            days: int = 300) -> pd.DataFrame:
    """
    從 Coinbase Advanced Trade API 取得 K 線資料
    不需要 API Key 即可讀取公開市場資料
    """
    end_ts   = int(datetime.utcnow().timestamp())
    start_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp())

    url = (f"https://api.coinbase.com/api/v3/brokerage/market/products/"
           f"{product_id}/candles"
           f"?start={start_ts}&end={end_ts}&granularity={granularity}")

    headers = {"Content-Type": "application/json"}
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json().get("candles", [])

    df = pd.DataFrame(data, columns=["start", "low", "high", "open", "close", "volume"])
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
    df["ema_short"] = calc_ema(df["close"], SHORT_EMA)
    df["ema_long"]  = calc_ema(df["close"], LONG_EMA)
    df["rsi"]       = calc_rsi(df["close"], RSI_PERIOD)
    df["macd"], df["macd_signal"], df["macd_hist"] = calc_macd(
        df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    # EMA 斜率（正 = 向上）
    df["ema_short_slope"] = df["ema_short"].diff()
    df["ema_long_slope"]  = df["ema_long"].diff()

    # 黃金/死亡交叉
    df["golden_cross"] = (df["ema_short"] > df["ema_long"]) & \
                         (df["ema_short"].shift(1) <= df["ema_long"].shift(1))
    df["death_cross"]  = (df["ema_short"] < df["ema_long"]) & \
                         (df["ema_short"].shift(1) >= df["ema_long"].shift(1))
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
        ema_s = row["ema_short"]
        ema_l = row["ema_long"]
        rsi   = row["rsi"]
        hist  = row["macd_hist"]

        equity = cash + holdings * price
        equity_curve.append({"date": str(row["date"].date()), "equity": round(equity, 2),
                              "price": round(price, 2)})

        both_up = row["ema_short_slope"] > 0 and row["ema_long_slope"] > 0

        # ── 賣出邏輯（優先）──────────────────────
        if holdings > 1e-8:
            # A: 死亡交叉 → 全賣
            if row["death_cross"]:
                sell(i, price, "死亡交叉", frac=1.0)
                half_sold = False
                continue

            # B: 跌破長線 3% → 全賣
            if price < ema_l * (1 - FULL_SELL_THRESHOLD):
                sell(i, price, f"跌破長線-{int(FULL_SELL_THRESHOLD*100)}%", frac=1.0)
                half_sold = False
                continue

            # C: 跌破短線 3% → 賣一半（若未執行過）
            if price < ema_s * (1 - HALF_SELL_THRESHOLD) and not half_sold:
                sell(i, price, f"跌破短線-{int(HALF_SELL_THRESHOLD*100)}%", frac=0.5)

        # ── 買入邏輯 ────────────────────────────
        buy_signal = (row["golden_cross"] and rsi < 70 and hist > 0)

        if buy_signal and holdings < 1e-8:
            buy(i, price, "黃金交叉+RSI+MACD", frac=1.0)

        # ── 再買回（半倉賣出後，反彈條件）─────
        if half_sold and both_up and price >= ema_l and rsi < 70 and hist > 0:
            buy(i, price, "長線反彈再買回", frac=1.0)  # 用剩餘現金買回

    # 最後收盤平倉
    if holdings > 1e-8:
        last = df.iloc[-1]
        sell(len(df)-1, last["close"], "回測結束平倉", frac=1.0)

    final_equity = cash
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    trades_df    = pd.DataFrame(trades)
    buy_hold     = (df.iloc[-1]["close"] / df.iloc[LONG_EMA]["close"] - 1) * 100

    # 最大回撤
    eq_vals = [e["equity"] for e in equity_curve]
    peak, max_dd = INITIAL_CAPITAL, 0.0
    for v in eq_vals:
        if v > peak: peak = v
        dd = (peak - v) / peak
        if dd > max_dd: max_dd = dd

    win_trades = sum(1 for t in trades if "SELL" in t["action"])
    total_fees = sum(t["fee"] for t in trades)

    return {
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return, 2),
        "buy_hold_return_pct": round(buy_hold, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_trades": len(trades),
        "total_fees_paid": round(total_fees, 2),
        "trades": trades,
        "equity_curve": equity_curve,
        "df": df
    }

def get_live_signal(product_id: str = "BTC-USD") -> dict:
    """取得即時交易訊號"""
    df     = fetch_coinbase_candles(product_id, days=200)
    df     = add_indicators(df).dropna().reset_index(drop=True)
    latest = df.iloc[-1]

    price  = latest["close"]
    ema_s  = latest["ema_short"]
    ema_l  = latest["ema_long"]
    rsi    = latest["rsi"]
    hist   = latest["macd_hist"]
    both_up = latest["ema_short_slope"] > 0 and latest["ema_long_slope"] > 0

    signal = "HOLD"
    reason = []

    if latest["golden_cross"] and rsi < 70 and hist > 0:
        signal = "BUY_ALL"
        reason.append(f"黃金交叉: EMA{SHORT_EMA}穿越EMA{LONG_EMA}")
    elif latest["death_cross"]:
        signal = "SELL_ALL"
        reason.append("死亡交叉")
    elif price < ema_l * (1 - FULL_SELL_THRESHOLD):
        signal = "SELL_ALL"
        reason.append(f"價格跌破長線{FULL_SELL_THRESHOLD*100:.0f}%")
    elif price < ema_s * (1 - HALF_SELL_THRESHOLD):
        signal = "SELL_HALF"
        reason.append(f"價格跌破短線{HALF_SELL_THRESHOLD*100:.0f}%")
    elif both_up and price >= ema_l and rsi < 70 and hist > 0:
        signal = "REBUY"
        reason.append("長線反彈再買回")

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
    print("=" * 55)
    print("  Coinbase 自動交易系統 (EMA+RSI+MACD)")
    print("=" * 55)
    print("  1. 執行歷史回測")
    print("  2. 啟動自動交易 (1 小時檢查一次)")
    print("  3. 啟動自動交易 (自訂頻率)")
    print("=" * 55)
    
    choice = input("請選擇功能 (1/2/3): ").strip()

    if choice == "1":
        print("\n[回測] 正在計算...")
        df = fetch_coinbase_candles("BTC-USD", days=300)
        result = run_backtest(df)
        print(f"\n策略報酬率: {result['total_return_pct']:+.2f}% | 交易次數: {result['total_trades']}")
        print("[完成] 回測結束。")

    elif choice == "2":
        run_auto_trading("BTC-USD", interval_seconds=3600)

    elif choice == "3":
        mins = input("輸入檢查頻率 (分鐘, 預設 60): ").strip()
        mins = int(mins) if mins else 60
        run_auto_trading("BTC-USD", interval_seconds=mins * 60)
        
    else:
        print("結束。")

