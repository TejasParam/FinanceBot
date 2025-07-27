# Clean Repository Structure

## Production Files

### Root Directory
- `test_world_class_system.py` - Main system test file
- `run_backtest.py` - Backtesting runner
- `data_collection.py` - Data fetching utilities
- `ml_predictor_enhanced.py` - Enhanced ML predictor with world-class features
- `news_sentiment_enhanced.py` - News sentiment analyzer
- `config.json` - Configuration file
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation
- `WORLD_CLASS_FEATURES.md` - Feature documentation

### Agents Directory (`agents/`)
Production-ready multi-agent system:
- `__init__.py` - Package initialization
- `base_agent.py` - Base agent class
- `coordinator.py` - Agent coordinator with portfolio optimization
- `fundamental_agent.py` - Fundamental analysis
- `intermarket_agent.py` - Cross-market correlation analysis
- `llm_agent.py` - LLM-based analysis
- `market_timing_agent.py` - Market timing with HFT signals
- `ml_agent.py` - ML predictions with transformers & RL
- `pattern_agent.py` - Pattern recognition
- `regime_agent.py` - Market regime detection
- `sentiment_agent.py` - Sentiment with alternative data
- `technical_agent.py` - Technical indicators
- `volatility_agent.py` - Volatility analysis

### Backtesting Directory (`backtesting/`)
- `backtest_runner.py` - Main backtest execution
- `historical_backtest.py` - Historical data backtesting
- `configs/` - Backtest configurations
- `results/` - Backtest results

### Model Storage (`models/`)
- Trained ML models and weights

### Backup Directory (`.backup/`)
All unused files have been moved here:
- `unused_files/` - Old scripts and demos
- `test_results/` - Previous test outputs

## Clean and Production-Ready

The repository is now streamlined with only essential production files. All experimental scripts, old demos, and test files have been archived in `.backup/` folder.