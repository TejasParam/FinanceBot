# Enhanced Finance Bot 🤖📈

A sophisticated, multi-agent finance analysis system with machine learning capabilities, backtesting, and comprehensive market analysis.

## 🚀 Features

### Core Analysis
- **Multi-Agent Architecture**: Coordinated analysis from technical, risk, news, and ML agents
- **Enhanced Portfolio Management**: Rule-based and ML-driven investment decisions
- **Real-time Data Integration**: Live market data, news sentiment, and technical indicators

### Advanced Capabilities
- **Machine Learning Predictions**: Trained models for price movement forecasting
- **Strategy Backtesting**: Historical performance evaluation with detailed metrics
- **Strategy Optimization**: Automated parameter tuning for better performance
- **Market Regime Analysis**: Dynamic market condition detection and adaptation
- **Risk Management**: Beta calculation, volatility analysis, and position sizing

### Technical Analysis
- RSI, MACD, Bollinger Bands, Moving Averages
- Fibonacci retracement levels
- Volume and momentum indicators
- Custom technical scoring algorithms

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd FinanceBot
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv finance_bot_env
   source finance_bot_env/bin/activate  # On Windows: finance_bot_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (create `.env` file):
   ```env
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
   ALPACA_API_KEY=your_alpaca_key
   ALPACA_SECRET_KEY=your_alpaca_secret
   ```

## 🏃 Quick Start

### Basic Usage
```python
from portfolio_manager_rule_based import AdvancedPortfolioManagerAgent

# Initialize the enhanced portfolio manager
manager = AdvancedPortfolioManagerAgent(use_ml=True)

# Run comprehensive analysis
result = manager.analyze_stock("AAPL")
print(f"Recommendation: {result['recommendation']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### Advanced Features
```python
# Train ML models
manager.train_ml_models("AAPL")

# Backtest strategy
backtest_result = manager.backtest_strategy("AAPL")

# Optimize strategy parameters
optimization_result = manager.optimize_strategy("AAPL")

# Analyze market regime
regime_result = manager.analyze_market_regime("AAPL")
```

## 🎯 Demo Scripts

### Full Feature Demo
```bash
python advanced_demo.py
```

### Quick Start Guide
```bash
python quick_start.py
```

## 📊 System Architecture

```
FinanceBot/
├── portfolio_manager_rule_based.py  # Main enhanced portfolio manager
├── ml_predictor.py                  # Machine learning pipeline
├── backtest_engine.py              # Strategy backtesting system
├── data_collection.py              # Market data retrieval
├── technical_analysis.py           # Technical indicators
├── risk_manager.py                 # Risk assessment
├── news_analyst.py                 # News sentiment analysis
├── market_coordinator.py           # Market coordination
└── models/                         # Trained ML models
```

## 🔧 Configuration

### Rule Weights (Optimizable)
```python
rule_weights = {
    'technical': 0.4,    # Technical analysis weight
    'sentiment': 0.2,    # News sentiment weight  
    'risk': 0.3,         # Risk assessment weight
    'ml_prediction': 0.1 # ML prediction weight
}
```

### ML Models
- **Random Forest**: Ensemble method for robust predictions
- **Gradient Boosting**: Advanced boosting for complex patterns
- **Logistic Regression**: Linear baseline model

## 📈 Performance Metrics

### Strategy Evaluation
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Excess Return**: Performance vs benchmark

### ML Model Metrics
- **Test Accuracy**: Out-of-sample prediction accuracy
- **Cross Validation**: Robust performance estimation
- **Feature Importance**: Key predictive factors

## 🔒 Risk Considerations

⚠️ **Important Disclaimers**:
- This system provides analysis, not guaranteed predictions
- Always conduct your own research before making investment decisions
- Past performance does not guarantee future results
- Use appropriate position sizing and risk management
- Consider market conditions and external factors

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the documentation in the code comments
- Run the demo scripts for examples

---

**Happy Trading! 📈🚀**