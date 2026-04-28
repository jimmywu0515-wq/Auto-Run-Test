# 🤖 Artemis Automated Trading System

An automated crypto trading and backtesting framework based on the Coinbase Advanced Trade API. This project compares **five different strategy dimensions** simultaneously and supports cross-platform deployment via Docker (ARM/x86).

---

## 🚀 Quick Start

Follow these steps to set up the environment:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/jimmywu0515-wq/Auto-Run-Test.git
    cd Autotrade
    ```

2.  **Configure API Keys**:
    Create a `.env` file and fill in your Coinbase API credentials:
    ```env
    COINBASE_API_KEY=your_key_name
    COINBASE_API_SECRET=your_private_key_content
    DRY_RUN=True   # Set to False for real trading
    ```

3.  **Setup Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

---

## 🕹️ Commands

### 1. Multi-Strategy Backtest
Compare all built-in strategies over a historical period.
*   **Default (300 days)**:
    ```bash
    ./venv/bin/python coinbase_strategy.py 1
    ```
*   **Custom Period (YTD)**:
    ```bash
    ./venv/bin/python coinbase_strategy.py 1 2026-01-01
    ```

### 2. Start Auto Trading (Dual EMA)
Checks signals every hour and executes trades based on the Dual EMA strategy.
```bash
./venv/bin/python coinbase_strategy.py 2
```

### 3. Start Auto Trading (Artemis RL)
Executes trades using the latest Artemis Triple-Barrier Reinforcement Learning model.
```bash
./venv/bin/python coinbase_strategy.py 3
```

---

## 🧠 Strategy Logic

The system integrates five distinct trading approaches:

### 1. Artemis Triple-Barrier RL (New) 🚀
*   **Logic**: Uses a PPO (Proximal Policy Optimization) model trained on 20 market features. It employs the **Triple-Barrier Method**:
    *   **Horizontal Take-Profit**: Set at 2.0x ATR.
    *   **Horizontal Stop-Loss**: Set at 2.0x ATR.
    *   **Vertical Timeout**: Force exit after 50 steps (approx. 50 days) if no other barrier is hit.
*   **Strength**: Adapts to volatile markets and manages risk dynamically through ATR-based barriers.

### 2. Dual EMA (20/50 + 5/10) 🏆
*   **Entry**: Uses long-term filters (20/50 EMA Golden Cross) to confirm the trend.
*   **Exit**: Uses sensitive short-term signals (5/10 EMA Death Cross) to lock in profits early.
*   **Strength**: Conservative entry with an aggressive exit strategy.

### 3. User's 20/50 MA Cross
*   **Logic**: Classic 20/50 MA crossover with a 1% price buffer to reduce noise and whip-saws.
*   **Strength**: Stable trend-following strategy.

### 4. Static Grid Trading
*   **Logic**: Places 10 buy/sell limit orders within a ±5% range of the starting price.
*   **Strength**: Profits from "sideways" or "range-bound" markets by buying low and selling high.

### 5. Artemis V2 (Legacy RL)
*   **Logic**: AI-driven model analyzing 18 features (RSI, MACD, Bollinger Bands, etc.) to decide position sizing (-100% to +100%).
*   **Strength**: Highly complex non-linear pattern recognition.

---

## 📊 Performance Report (YTD 2026)
**Period**: 2026-01-01 to 2026-04-28

| Strategy Name           | Return   | Max DD   | Trades |
| :---------------------- | :------- | :------- | :----- |
| Dual EMA (20/50+5/10)   | -8.64%   | 8.64%    | 2      |
| 20/50 MA Cross          | -9.57%   | 25.19%   | 27     |
| Static Grid             | -15.04%  | 31.87%   | 49     |
| Artemis V2 (Legacy)     | -17.25%  | 25.28%   | 22     |
| **Artemis Triple-Barrier**| **+2.08%** | **14.40%** | **4**    |
| Buy & Hold (BTC)        | -13.42%  | N/A      | 0      |

---

## 🐳 Docker Deployment

For 24/7 operation on NAS or servers:
```bash
docker-compose up --build -d
```

---

## ⚠️ Disclaimer
Cryptocurrency trading involves significant risk. This software is for educational purposes only. The developers do not guarantee any profits. Always use `DRY_RUN=True` before committing real capital.
