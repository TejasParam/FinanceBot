# Running the Alpaca Trading Bot

Your bot is now configured and ready to trade! It's using your Alpaca paper trading credentials from the .env file.

## Current Status
- ✅ Account connected: $100,000 paper trading balance
- ✅ API credentials loaded from .env
- ✅ All systems operational

## Quick Commands

### 1. Test Run (Immediate, One-Time)
```bash
python main_trading_bot.py --test
```
This runs once to verify everything works.

### 2. Live Trading Mode (Continuous)
```bash
python main_trading_bot.py
```
This will:
- Run continuously during market hours (9:30 AM - 4:00 PM ET)
- Generate trading signals every 15 minutes
- Execute trades automatically
- Apply risk management (5% stop-loss, 10% take-profit)
- Rebalance portfolio daily

### 3. Start Dashboard
In a separate terminal:
```bash
python dashboard_api.py
```
Then open `dashboard.html` in your browser to monitor trades.

## What Happens During Market Hours

The bot will:
1. **Analyze these 30 stocks**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, JNJ, WMT, PG, UNH, HD, MA, DIS, PYPL, NFLX, ADBE, CRM, PFE, TMO, ABBV, NKE, COST, CVX, WFC, MCD, LLY, ACN

2. **Make trading decisions based on**:
   - ML predictions (70% weight)
   - News sentiment (30% weight)
   - Technical indicators (RSI, MACD)
   - Risk metrics

3. **Execute trades when**:
   - Combined confidence > 60%
   - Risk limits not exceeded
   - Position size: 2-10% of portfolio

## Monitor Your Bot

### Check Logs
```bash
tail -f trading_bot.log
```

### View Reports
Daily reports are saved in `reports/` folder:
```bash
ls -la reports/
cat reports/daily_report_20250714.json | jq .
```

### Alpaca Dashboard
You can also monitor trades at: https://app.alpaca.markets

## Trading Schedule

- **Pre-market**: 9:15 AM - Health check
- **Market hours**: 9:30 AM - 4:00 PM - Active trading
- **Near close**: 3:45 PM - Portfolio rebalancing  
- **After market**: 4:30 PM - Daily report generation

## Stop the Bot

Press `Ctrl+C` to stop. The bot will:
- Cancel all open orders
- Generate final report
- Shut down gracefully

## Notes

- Currently using **paper trading** (fake money)
- No risk to real funds
- Great for testing strategies
- All trades visible in Alpaca dashboard

Happy automated trading! 🚀