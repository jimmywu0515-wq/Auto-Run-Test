# Coinbase Automated Trading Strategy

This project implements an automated trading strategy for Coinbase Advanced Trade using EMA, RSI, and MACD indicators.

## Setup Instructions

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd auto-test
    ```

2.  **Create a `.env` file**:
    Copy `.env.example` to `.env` and fill in your Coinbase API credentials.
    ```bash
    cp .env.example .env
    ```

3.  **Install dependencies**:
    ```bash
    pip install pandas numpy requests coinbase-advanced-py python-dotenv
    ```

4.  **Run the strategy**:
    ```bash
    python coinbase_strategy.py
    ```

## Strategy Details
- **Indicators**: EMA 20/50, RSI 14, MACD 12/26/9.
- **Rules**:
  - Buy when EMA20 crosses above EMA50, RSI < 70, and MACD histogram > 0.
  - Sell all when EMA20 crosses below EMA50 or price drops 3% below EMA50.
  - Sell half when price drops 3% below EMA20.

## Disclaimer
Trading cryptocurrencies involves significant risk. This bot is for educational purposes only. Use it at your own risk.
