# Quick Start Guide - Alpaca Trading Bot

## Step 1: Install Dependencies

```bash
# Install the required packages
./install_alpaca_dependencies.sh

# Or manually install:
pip install alpaca-py schedule flask flask-cors textblob vaderSentiment
```

## Step 2: Set Up Configuration

Create a `config.json` file with your Alpaca API credentials:

```json
{
    "alpaca_api_key": "YOUR_PAPER_TRADING_API_KEY",
    "alpaca_secret_key": "YOUR_PAPER_TRADING_SECRET_KEY",
    "initial_capital": 100000,
    "trading_hours": {
        "start": "09:30",
        "end": "16:00"  
    },
    "rebalance_frequency": "daily",
    "risk_check_interval_minutes": 5,
    "signal_generation_interval_minutes": 15
}
```

Get your FREE Alpaca paper trading API keys:
1. Go to https://alpaca.markets
2. Sign up for a free account
3. Navigate to "Your API Keys" 
4. Generate Paper Trading API keys
5. Copy them to config.json

## Step 3: Run the Trading Bot

Terminal 1 - Start the trading bot:
```bash
python main_trading_bot.py
```

Terminal 2 - Start the dashboard API:
```bash
python dashboard_api.py
```

Terminal 3 - Open the dashboard:
```bash
# Serve the dashboard
python -m http.server 8080

# Then open in browser:
# http://localhost:8080/dashboard.html
```

## Step 4: Monitor Your Portfolio

The dashboard shows:
- Portfolio value and P&L
- Current positions
- Trading signals with ML confidence scores
- Trade history with reasons
- Performance metrics

## Trading Strategy

The bot will:
1. Analyze 30 major stocks every 15 minutes
2. Use ML models (55% accuracy) + sentiment analysis
3. Buy when confidence > 60%
4. Apply 5% stop-loss and 10% take-profit
5. Rebalance portfolio daily
6. Maximum 20 positions, 2-10% each

## Important Notes

- This uses PAPER TRADING (fake money) - perfect for testing
- Only trades during market hours (9:30 AM - 4:00 PM ET)
- All trades are logged with detailed reasoning
- Check `trading_bot.log` for detailed logs

## Troubleshooting

If you get import errors:
```bash
pip install -r requirements.txt
```

If ML models aren't loaded:
- The bot will train new models on first run
- Or use the existing models in `models/portfolio_models.pkl`

## Stop the Bot

Press `Ctrl+C` in the terminal running the bot. It will:
- Cancel all open orders
- Generate a final report
- Shut down gracefully