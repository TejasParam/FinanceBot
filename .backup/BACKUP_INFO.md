# Backup Folder Contents

This folder contains files that are not actively used by the current agentic AI finance bot system. These files have been moved here to keep the main project directory clean and organized.

## Legacy/Non-Agentic Files

### Core Legacy Files
- `portfolio_manager.py` - Original non-agentic portfolio manager
- `market_coordinator.py` - Original non-agentic market coordinator
- `market_coordinator_rule_based.py` - Rule-based market coordinator
- `quick_start.py` - Original quick start script (replaced by advanced_demo.py)

### Standalone Analysis Scripts
- `ml_predictor.py` - Standalone ML prediction script
- `news_analyst.py` - Standalone news analysis script
- `risk_manager.py` - Standalone risk management script
- `complete_market_analysis.py` - Standalone market analysis (now in demos/)
- `market_scanner.py` - Standalone market scanner (now in demos/)
- `portfolio_builder.py` - Standalone portfolio builder (now in demos/)

### Trading Integration
- `alpaca_trader.py` - Alpaca trading integration (optional)

## Current Active System

The main agentic system uses:
- `agentic_portfolio_manager.py` - Main entry point
- `agents/` folder - All specialized AI agents
- `advanced_demo.py` - Main demonstration script
- `demo_local_llm.py` - LLM integration demo
- `demos/` folder - Organized demo scripts

## Restoration

If you need any of these backup files, simply move them back to the main directory:
```bash
mv .backup/filename.py ./
```

## Purpose

These files were moved to backup to:
1. Clean up the main directory structure
2. Focus on the current agentic AI architecture
3. Preserve legacy code for reference
4. Improve project organization and maintainability

Last updated: July 13, 2025
