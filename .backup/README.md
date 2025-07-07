# FinanceBot Legacy Files Backup

This folder contains legacy and unused files from the FinanceBot project that have been superseded by the new agentic architecture.

## Files in Backup

### Legacy Portfolio Managers
- `portfolio_manager.py` - Original LLM-based portfolio manager (superseded by agentic system)
- `portfolio_manager_rule_based.py` - Rule-based portfolio manager (integrated into agentic system)

### Legacy Coordinators
- `market_coordinator.py` - Original market coordinator (superseded by AgentCoordinator)
- `market_coordinator_rule_based.py` - Rule-based coordinator (superseded by agentic system)

### Standalone Tools
- `complete_market_analysis.py` - Comprehensive market analysis script (functionality integrated)
- `market_scanner.py` - Market scanning functionality (can be re-integrated if needed)
- `portfolio_builder.py` - Portfolio building tools (functionality integrated)
- `quick_start.py` - Original quick start script (superseded by new demos)

### Trading Integration
- `alpaca_trader.py` - Alpaca trading API integration (not currently used)

## Why These Files Were Moved

These files were moved to backup because:

1. **Superseded by Agentic Architecture**: The new multi-agent system provides all functionality with better organization and capabilities
2. **Legacy Code**: These files use older patterns and architectures that are no longer the primary approach
3. **Reduced Complexity**: Moving them simplifies the main project structure and reduces confusion
4. **Preserved for Reference**: Files are preserved in case any functionality needs to be extracted or referenced

## Active System Files

The current active FinanceBot system uses:
- `agentic_portfolio_manager.py` - Main agentic system interface
- `agents/` - All specialized AI agents
- `ml_predictor.py` - Machine learning functionality
- `backtest_engine.py` - Strategy backtesting
- `data_collection.py` - Market data collection
- `technical_analysis.py` - Technical analysis functions
- `risk_manager.py` - Risk management utilities
- `news_analyst.py` - News sentiment analysis
- Demo files: `advanced_demo.py`, `demo_local_llm.py`, etc.

## Restoring Files

If you need to restore any of these files to the main directory:

```bash
cd /Users/tanis/Documents/FinanceBot
mv .backup/filename.py ./
```

## Note

These files are functional and were working when moved to backup. They represent the evolution of the FinanceBot system and could potentially be useful for specific use cases or as reference material.
