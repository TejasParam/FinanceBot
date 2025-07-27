# Accuracy Improvement Plan

## Current Issues
- System achieves ~40-45% accuracy instead of 80%
- All agents weighted equally regardless of performance
- Trading in all market conditions
- No position sizing optimization

## Improvement Strategies

### 1. Weighted Ensemble Voting
Instead of simple averaging, weight agents by their historical accuracy:

```python
# Track agent performance
agent_performance = {
    'technical': {'correct': 0, 'total': 0, 'weight': 1.0},
    'fundamental': {'correct': 0, 'total': 0, 'weight': 1.0},
    'sentiment': {'correct': 0, 'total': 0, 'weight': 1.0},
    # etc...
}

# Update weights based on rolling accuracy
def update_agent_weights(agent_name, was_correct):
    perf = agent_performance[agent_name]
    perf['total'] += 1
    if was_correct:
        perf['correct'] += 1
    
    # Calculate rolling accuracy
    if perf['total'] > 20:  # Minimum samples
        accuracy = perf['correct'] / perf['total']
        # Weight = accuracy^2 to emphasize better agents
        perf['weight'] = accuracy ** 2
```

### 2. Market Regime Filtering
Only trade when market conditions are favorable:

```python
def should_trade(market_regime):
    # Avoid trading in:
    # - High volatility periods (VIX > 30)
    # - Major news events
    # - Low liquidity periods
    # - Unclear trends
    
    favorable_regimes = ['trending_up', 'trending_down', 'low_volatility']
    return market_regime in favorable_regimes
```

### 3. Dynamic Confidence Thresholds
Adjust confidence requirements based on market conditions:

```python
def get_confidence_threshold(volatility, market_trend):
    base_threshold = 0.70
    
    # Require higher confidence in volatile markets
    if volatility > 0.02:  # 2% daily volatility
        base_threshold += 0.10
    
    # Require higher confidence in sideways markets
    if market_trend == 'sideways':
        base_threshold += 0.05
    
    return base_threshold
```

### 4. Feature Engineering for ML
Add more predictive features:

```python
# Technical features
- Relative volume (volume / avg_volume)
- Price distance from key levels
- Momentum divergence indicators
- Market breadth indicators

# Microstructure features  
- Order book imbalance
- Bid-ask spread changes
- Large trade detection
- Dark pool activity estimation

# Intermarket features
- Sector relative strength
- Currency impacts
- Bond-equity correlation
- Commodity trends
```

### 5. Signal Combination Rules
Instead of simple averaging, use rules:

```python
def combine_signals(agent_scores):
    # Require agreement from multiple agent types
    technical_bullish = any(s > 0.5 for s in technical_scores)
    fundamental_bullish = fundamental_score > 0.3
    sentiment_bullish = sentiment_score > 0.4
    
    # Strong buy only if 2+ categories agree
    if sum([technical_bullish, fundamental_bullish, sentiment_bullish]) >= 2:
        return 'BUY'
    
    # Strong sell if 2+ categories bearish
    # ... similar logic
```

### 6. Time-Based Filters
Avoid problematic periods:

```python
def is_good_trading_time(timestamp):
    # Avoid:
    # - First/last 30 minutes of trading day
    # - Days before major events (FOMC, earnings)
    # - Triple witching days
    # - Holiday weeks
    
    hour = timestamp.hour
    if hour < 10 or hour > 15:  # Avoid open/close
        return False
    
    # Check economic calendar
    if has_major_event_today():
        return False
    
    return True
```

### 7. Position Sizing with Kelly Criterion
Already implemented but can be enhanced:

```python
def calculate_position_size(win_probability, avg_win, avg_loss):
    # Kelly formula: f = (p*b - q) / b
    # where p = win prob, q = loss prob, b = win/loss ratio
    
    # Add safety factor
    kelly_fraction = calculate_kelly(win_probability, avg_win, avg_loss)
    safe_fraction = kelly_fraction * 0.25  # Use 25% of Kelly
    
    # Cap at maximum position size
    return min(safe_fraction, 0.10)  # Max 10% per position
```

### 8. Confirmation Requirements
Require multiple timeframe confirmation:

```python
def get_multi_timeframe_confirmation(ticker):
    # Check daily, weekly, monthly trends
    daily_trend = analyze_trend(ticker, '1d')
    weekly_trend = analyze_trend(ticker, '1wk') 
    monthly_trend = analyze_trend(ticker, '1mo')
    
    # All timeframes must agree
    if daily_trend == weekly_trend == monthly_trend:
        return True, 1.0  # High confidence
    elif daily_trend == weekly_trend:
        return True, 0.7  # Medium confidence
    else:
        return False, 0.0  # No trade
```

### 9. Stop-Loss and Take-Profit Rules
Implement systematic exit rules:

```python
def calculate_exit_levels(entry_price, volatility, trend_strength):
    # Dynamic stop-loss based on volatility
    stop_loss = entry_price * (1 - 2 * volatility)
    
    # Take profit based on risk-reward ratio
    take_profit = entry_price * (1 + 3 * volatility)  # 3:1 reward:risk
    
    # Trail stop after reaching 1:1
    trail_activation = entry_price * (1 + volatility)
    
    return stop_loss, take_profit, trail_activation
```

### 10. Learning from Mistakes
Track why predictions fail:

```python
def analyze_failed_prediction(prediction, actual):
    failure_reasons = []
    
    # Was it a regime change?
    if market_regime_changed(prediction.date):
        failure_reasons.append('regime_change')
    
    # Was it news-driven?
    if had_unexpected_news(prediction.symbol, prediction.date):
        failure_reasons.append('news_event')
    
    # Was it a false signal?
    if prediction.confidence < 0.75:
        failure_reasons.append('low_confidence')
    
    # Update agent weights based on failure analysis
    adjust_agent_weights_for_failure(failure_reasons)
```

## Implementation Priority

1. **High Priority** (Biggest impact):
   - Weighted ensemble voting
   - Market regime filtering
   - Dynamic confidence thresholds

2. **Medium Priority**:
   - Feature engineering
   - Time-based filters
   - Multi-timeframe confirmation

3. **Lower Priority** (Refinements):
   - Advanced position sizing
   - Failure analysis system
   - Stop-loss optimization

## Expected Improvement

With these enhancements, we could potentially improve accuracy from 40-45% to:
- **60-65%** with basic improvements (1-3)
- **70-75%** with full implementation (1-6)
- **75-80%** with continuous learning and optimization (all)

The key is to:
1. Trade less frequently but with higher conviction
2. Weight successful agents more heavily
3. Avoid unfavorable market conditions
4. Use proper risk management
5. Learn from failures and adapt