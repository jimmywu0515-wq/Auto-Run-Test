# 🤖 Coinbase Pro 自動化交易跨策略系統

這是一個基於 Coinbase Advanced Trade API 的自動化交易與回測框架。本專案能讓您在同一時間對比 **四種不同維度** 的交易策略效能，並支援 Docker 跨平台部署（ARM/x86）。

---

## 🚀 快速開始 (安裝步驟)

請按照以下步驟在您的電腦上建立運行環境：

1.  **下載專案**：
    ```bash
    git clone https://github.com/jimmywu0515-wq/Auto-Run-Test.git
    cd Autotrade
    ```

2.  **設定 API 金鑰**：
    建立一個名為 `.env` 的檔案，並填入您的 Coinbase API 資訊：
    ```env
    COINBASE_API_KEY=您的Key名稱
    COINBASE_API_SECRET=您的私鑰內容
    DRY_RUN=True   # True 為模擬交易，False 為真錢交易
    ```

3.  **建立虛擬環境與安裝套件**：
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Mac / Linux 指令
    pip install -r requirements.txt
    ```

---

## 🕹️ 指令說明 (如何執行)

執行程式時，您可以直接在指令後方加上數字來選擇功能：

### 選項 1：多策略歷史回測
回測會同時執行四種策略，並給出比較報表。
*   **預設回測 (300天)**：
    ```bash
    ./venv/bin/python coinbase_strategy.py 1
    ```
*   **自訂回測時段 (天數或日期)**：
    ```bash
    ./venv/bin/python coinbase_strategy.py 1 500         # 回測過去 500 天
    ./venv/bin/python coinbase_strategy.py 1 2025-01-01  # 從 2025 年元旦開始回測
    ```

### 選項 2：啟動自動交易 (每小時檢查)
機器人會每小時抓取一次訊號，若符合 **Dual EMA** 策略則自動下單。
```bash
./venv/bin/python coinbase_strategy.py 2
```

### 選項 3：啟動自動交易 (自訂頻率)
例如每 30 分鐘檢查一次：
```bash
./venv/bin/python coinbase_strategy.py 3 30
```

---

## 🧠 策略邏輯詳解

本系統內建四套戰力各異的策略：

1.  **Dual EMA (20/50 + 5/10)** 🏆
    *   **進場**：看長線（20/50 均線黃金交叉），確保大趨勢向上。
    *   **退場**：看短線（5/10 均線死亡交叉），在轉折初期就逃跑，保住獲利。
    *   **特色**：進場嚴謹、逃跑神速。

2.  **User's 5/10 MA Cross**
    *   **邏輯**：經典 5/10 均線交叉，搭配 1% 的價格緩衝區（Buffer）來避免頻繁交易。
    *   **特色**：傳統穩定策略，適合波動明顯的波段。

3.  **Static Grid (網格交易)**
    *   **邏輯**：在當前價格上下 5% 建立 10 層網格。
    *   **特色**：低買高賣，最適合「橫盤震盪」的行情。

4.  **Artemis V2 (AI 強化學習)** 🤖
    *   **邏輯**：由 AI 驅動，分析 18 種市場特徵（RSI、MACD、布林帶等）自行決定倉位。
    *   **特色**：科技感最強，適合捕捉非長規的市場規律。

---

## 🐳 Docker 部署 (推薦)

如果您希望在 NAS、雲端伺服器或 Raspberry Pi 上 24 小時運行，建議使用 Docker：

*   **啟動服務**：
    ```bash
    docker-compose up --build
    ```
*   **注意事項**：請確保您的 `.env` 檔案已正確設定，Docker 會自動讀取裡面的金鑰。

---

## ⚠️ 免責聲明
加密貨幣交易具有高風險。本程式僅供技術交流與教學使用，開發者不保證任何獲利。請在投入真實資金前，先進行充分的模擬測試 (DRY_RUN=True)。
