# 使用支援多平台的 Python 基礎映像檔
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統相依套件 (編譯某些 Python 套件所需)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 複製需求文件並安裝 Python 套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製其餘程式碼
COPY . .

# 設定環境變數 (確保 Python 輸出即時顯示)
ENV PYTHONUNBUFFERED=1

# 預設執行回測展示
CMD ["python", "coinbase_strategy.py", "1", "100"]
