# FinanceBot Quick Start Guide

## Overview
FinanceBot is an advanced AI-powered financial analysis system that uses multiple specialized agents to analyze stocks, manage portfolios, and make data-driven investment decisions.

## Core Features
- **Multi-Agent Analysis**: Technical, fundamental, sentiment, and ML-based analysis
- **Portfolio Optimization**: Advanced strategies including Mean-Variance, Black-Litterman, and Risk Parity
- **Risk Management**: Comprehensive VaR, Monte Carlo simulations, and stress testing
- **Real-time Processing**: WebSocket support and streaming predictions
- **Backtesting**: Walk-forward analysis with realistic transaction costs

## Quick Start

### 1. Basic Stock Analysis
```python
from agentic_portfolio_manager import AgenticPortfolioManager

# Initialize the manager
manager = AgenticPortfolioManager()

# Analyze a single stock
analysis = manager.analyze_stock("AAPL")
print(analysis['summary'])
```

### 2. Portfolio Creation
```python
from portfolio_optimizer import EnhancedPortfolioOptimizer

# Create optimizer
optimizer = EnhancedPortfolioOptimizer()

# Optimize portfolio
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
result = optimizer.optimize_portfolio(
    tickers=tickers,
    strategy='mean_variance'
)

print("Optimal weights:", result['weights'])
```

### 3. Risk Analysis
```python
from risk_manager_enhanced import EnhancedRiskManager

# Initialize risk manager
risk_mgr = EnhancedRiskManager()

# Calculate VaR
var_metrics = risk_mgr.calculate_all_var_metrics("AAPL")
print(f"VaR (95%): {var_metrics['var_historical']:.2f}%")

# Run Monte Carlo simulation
portfolio = {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}
mc_results = risk_mgr.monte_carlo_portfolio_simulation(portfolio)
```

### 4. Run Demos
```bash
# Complete market analysis
python demos/complete_market_analysis.py

# Multi-stock scanner
python demos/market_scanner.py

# Portfolio builder
python demos/portfolio_builder.py
```

## Project Structure

### Core Modules
- `portfolio_optimizer.py` - Advanced portfolio optimization
- `risk_manager_enhanced.py` - Risk analysis and management
- `backtest_engine_enhanced.py` - Backtesting with walk-forward analysis
- `data_validation_pipeline.py` - Data quality assurance
- `parallel_agent_executor.py` - Parallel processing framework

### Agent System
- `agents/technical_agent.py` - Technical analysis
- `agents/fundamental_agent.py` - Fundamental analysis
- `agents/sentiment_agent.py` - News sentiment analysis
- `agents/ml_agent_enhanced.py` - Advanced ML predictions
- `agents/regime_agent.py` - Market regime detection

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Optional advanced features:
```bash
pip install pypfopt cvxpy ray
```

## Configuration

The system works out of the box with default settings. For advanced configuration:

1. Risk tolerance in portfolio optimization
2. Transaction cost models in backtesting
3. ML model parameters
4. Real-time data sources

## Next Steps

1. Read `AGENTIC_AI_GUIDE.md` for detailed agent system documentation
2. Explore the demos folder for example implementations
3. Check the backup folder for additional examples and test files

## Support

For issues or questions:
- Check the documentation in the project
- Review example code in demos/
- Examine test files in .backup/test_files/ for usage examples