# AlpacaBot - AI-Powered Automated Trading System

An intelligent portfolio management system that uses machine learning, sentiment analysis, and technical indicators to automatically trade on Alpaca's paper trading platform.

## 🚀 Features

- **Machine Learning Predictions**: Uses ensemble models (Random Forest, XGBoost, LightGBM) achieving ~55% accuracy
- **Sentiment Analysis**: Analyzes news and social media sentiment for trading decisions
- **Risk Management**: Automated stop-loss, take-profit, and position sizing
- **Portfolio Optimization**: Dynamic rebalancing based on Modern Portfolio Theory
- **Real-time Dashboard**: Web-based dashboard showing positions, trades, and performance
- **Automated Trading**: Runs continuously during market hours with configurable intervals

## 📋 Prerequisites

- Python 3.8+
- Alpaca Paper Trading Account (free at https://alpaca.markets)
- API Keys from Alpaca

## 🛠️ Installation

1. Clone the repository:
```bash
git clone [your-repo]
cd FinanceBot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional system dependencies:
```bash
# For macOS (required for XGBoost)
brew install libomp

# For TA-Lib (optional but recommended)
# macOS: brew install ta-lib
# Ubuntu: sudo apt-get install ta-lib
```

## ⚙️ Configuration

1. Create a `config.json` file:
```json
{
    "alpaca_api_key": "YOUR_ALPACA_API_KEY",
    "alpaca_secret_key": "YOUR_ALPACA_SECRET_KEY",
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

2. Get your Alpaca API keys:
   - Sign up at https://alpaca.markets
   - Go to Your API Keys in the dashboard
   - Generate new keys for paper trading
   - Copy them to your config.json

## 🏃 Running the Bot

### 1. Start the Trading Bot:
```bash
python main_trading_bot.py
```

The bot will:
- Generate initial portfolio based on ML predictions
- Monitor positions and execute trades
- Rebalance portfolio daily
- Apply risk management (stop-loss at 5%, take-profit at 10%)

### 2. Start the Dashboard API:
```bash
python dashboard_api.py
```

### 3. Open the Dashboard:
Open `dashboard.html` in your browser or serve it with:
```bash
python -m http.server 8080
```
Then navigate to http://localhost:8080/dashboard.html

## 📊 How It Works

### Trading Logic

1. **Signal Generation** (every 15 minutes):
   - Analyzes 30 stocks from major indices
   - ML models predict price direction (up/down)
   - Sentiment analysis on news/social media
   - Technical indicators confirmation
   - Combines signals with weights: 70% ML, 30% sentiment

2. **Position Sizing**:
   - Uses Kelly Criterion modified by volatility
   - Min position: 2% of portfolio
   - Max position: 10% of portfolio
   - Max 20 total positions

3. **Risk Management**:
   - Stop-loss: -5% from entry
   - Take-profit: +10% from entry
   - Maximum portfolio exposure: 95%
   - Daily Value at Risk limit: 2%

### ML Models

The system uses ensemble learning with:
- Random Forest (100 trees)
- XGBoost (300 estimators)
- LightGBM (300 estimators)
- Neural Network (3 layers)
- LSTM (if TensorFlow available)

Features include:
- 100+ technical indicators
- Price momentum and volatility
- Volume patterns
- Market microstructure

## 📈 Performance Metrics

The dashboard displays:
- Portfolio value and returns
- Individual position P&L
- Trading signals with confidence scores
- Win rate and Sharpe ratio
- Risk metrics (VaR, volatility)

## 🔧 Customization

### Modify Trading Universe
Edit `trading_universe` in `automated_portfolio_manager.py`:
```python
self.trading_universe = [
    'AAPL', 'MSFT', 'GOOGL', # Add your stocks
]
```

### Adjust Risk Parameters
In `automated_portfolio_manager.py`:
```python
self.min_confidence_threshold = 0.60  # Min confidence for trading
self.stop_loss_pct = 0.05  # Stop loss percentage
self.take_profit_pct = 0.10  # Take profit percentage
```

### Change ML Models
The models can be retrained with:
```bash
python train_models.py --symbols AAPL,MSFT,GOOGL --period 2y
```

## 📁 Project Structure

```
FinanceBot/
├── main_trading_bot.py          # Main trading loop
├── alpaca_integration.py        # Alpaca API wrapper
├── automated_portfolio_manager.py # Portfolio management logic
├── ml_predictor_enhanced.py     # ML prediction models
├── sentiment_analyzer.py        # Sentiment analysis
├── dashboard_api.py            # Flask API for dashboard
├── dashboard.html              # Web dashboard
├── config.json                 # Configuration (create this)
├── models/                     # Trained ML models
├── reports/                    # Daily portfolio reports
├── trades_log.json            # Trade history
└── execution_log.json         # Signal execution log
```

## ⚠️ Important Notes

1. **Paper Trading Only**: This bot is configured for paper trading. Never use with real money without extensive testing.

2. **Market Hours**: The bot only trades during market hours (9:30 AM - 4:00 PM ET, weekdays).

3. **API Rate Limits**: Alpaca has rate limits. The bot is configured to respect these.

4. **Risk Disclaimer**: Trading involves risk. Past performance doesn't guarantee future results.

## 🐛 Troubleshooting

### Common Issues:

1. **"No module named 'alpaca'"**:
   ```bash
   pip install alpaca-py
   ```

2. **"XGBoost Library could not be loaded"**:
   ```bash
   # macOS
   brew install libomp
   ```

3. **"Trading is blocked"**:
   - Check your Alpaca account status
   - Ensure you're using paper trading API keys

### Logs

Check `trading_bot.log` for detailed execution logs.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is for educational purposes. Use at your own risk.