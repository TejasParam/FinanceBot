"""
Automated Portfolio Manager
Integrates ML predictions, sentiment analysis, and risk management for automated trading
"""

import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json

from alpaca_integration import AlpacaTradingBot
from ml_predictor_enhanced import EnhancedMLPredictor
from sentiment_analyzer import MarketSentimentAnalyzer
from data_collection import DataCollectionAgent
from portfolio_optimizer import EnhancedPortfolioOptimizer

@dataclass
class TradingSignal:
    """Trading signal with all relevant information"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    ml_score: float
    sentiment_score: float
    risk_score: float
    volatility: float
    position_size: int
    reasons: List[str]
    timestamp: datetime

class AutomatedPortfolioManager:
    """Main class for automated portfolio management"""
    
    def __init__(self, alpaca_bot: AlpacaTradingBot, initial_capital: float = 100000):
        """
        Initialize the automated portfolio manager
        
        Args:
            alpaca_bot: Initialized AlpacaTradingBot instance
            initial_capital: Initial capital (for backtesting/tracking)
        """
        self.alpaca_bot = alpaca_bot
        self.initial_capital = initial_capital
        
        # Initialize components
        self.ml_predictor = EnhancedMLPredictor()
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        self.data_collector = DataCollectionAgent()
        self.portfolio_optimizer = EnhancedPortfolioOptimizer()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Trading parameters
        self.min_confidence_threshold = 0.60  # Minimum confidence for trading
        self.max_portfolio_exposure = 0.95  # Max 95% invested
        self.rebalance_threshold = 0.10  # Rebalance if weight drift > 10%
        self.sentiment_weight = 0.30  # Weight given to sentiment in decisions
        self.ml_weight = 0.70  # Weight given to ML predictions
        
        # Risk parameters
        self.max_sector_exposure = 0.30  # Max 30% in one sector
        self.max_correlation = 0.70  # Max correlation between positions
        self.var_limit = 0.02  # Max 2% daily VaR
        
        # Universe of stocks to trade
        self.trading_universe = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'MA',
            'DIS', 'PYPL', 'NFLX', 'ADBE', 'CRM', 'PFE', 'TMO',
            'ABBV', 'NKE', 'COST', 'CVX', 'WFC', 'MCD', 'LLY', 'ACN'
        ]
        
        # Load ML models
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained ML models"""
        try:
            self.ml_predictor.load_models('models/portfolio_models.pkl')
            self.logger.info("ML models loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load ML models: {e}")
            self.logger.info("Will train new models on first run")
    
    def generate_trading_signals(self) -> List[TradingSignal]:
        """Generate trading signals for all stocks in universe"""
        signals = []
        account_info = self.alpaca_bot.get_account_info()
        portfolio_value = account_info['portfolio_value']
        
        for symbol in self.trading_universe:
            try:
                signal = self._analyze_stock(symbol, portfolio_value)
                if signal:
                    signals.append(signal)
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                
        return signals
    
    def _analyze_stock(self, symbol: str, portfolio_value: float) -> Optional[TradingSignal]:
        """Analyze a single stock and generate trading signal"""
        reasons = []
        
        # Get historical data
        data = self.data_collector.fetch_stock_data(symbol, period='6mo')
        if data is None or len(data) < 50:
            return None
        
        # ML prediction
        ml_prediction = self._get_ml_prediction(symbol, data)
        ml_score = ml_prediction['probability_up']
        ml_confidence = ml_prediction['confidence']
        
        # Sentiment analysis
        sentiment = self.sentiment_analyzer.analyze_stock_sentiment(symbol)
        sentiment_score = sentiment['compound_score']
        
        # Technical analysis
        technical_signals = self._analyze_technicals(data)
        
        # Risk metrics
        volatility = self._calculate_volatility(data)
        risk_score = self._calculate_risk_score(symbol, volatility)
        
        # Combine signals
        combined_score = (
            self.ml_weight * ml_score + 
            self.sentiment_weight * (sentiment_score + 1) / 2  # Normalize to 0-1
        )
        
        # Add technical confirmation
        if technical_signals['rsi'] < 30:
            combined_score *= 1.1
            reasons.append("RSI oversold")
        elif technical_signals['rsi'] > 70:
            combined_score *= 0.9
            reasons.append("RSI overbought")
            
        if technical_signals['macd_signal'] == 'bullish':
            combined_score *= 1.05
            reasons.append("MACD bullish crossover")
        elif technical_signals['macd_signal'] == 'bearish':
            combined_score *= 0.95
            reasons.append("MACD bearish crossover")
        
        # Determine action
        current_position = self.alpaca_bot.get_position_value(symbol)
        confidence = ml_confidence * (1 - risk_score)  # Adjust for risk
        
        if combined_score > 0.60 and confidence > self.min_confidence_threshold:
            if current_position == 0:
                action = 'BUY'
                reasons.append(f"ML prediction: {ml_score:.1%}")
                reasons.append(f"Sentiment: {sentiment['sentiment']}")
                position_size = self.alpaca_bot.calculate_position_size(
                    symbol, confidence, volatility, portfolio_value
                )
            else:
                action = 'HOLD'
                position_size = 0
                reasons.append("Already in position")
        elif combined_score < 0.40 and current_position > 0:
            action = 'SELL'
            reasons.append(f"ML prediction: {ml_score:.1%}")
            reasons.append(f"Negative outlook")
            position_size = 0
        else:
            action = 'HOLD'
            position_size = 0
            if current_position > 0:
                reasons.append("Maintaining position")
            else:
                reasons.append("Insufficient confidence")
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            ml_score=ml_score,
            sentiment_score=sentiment_score,
            risk_score=risk_score,
            volatility=volatility,
            position_size=position_size,
            reasons=reasons,
            timestamp=datetime.now()
        )
    
    def _get_ml_prediction(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Get ML prediction for a stock"""
        try:
            # Prepare features
            features_df = self.ml_predictor.prepare_enhanced_features(data)
            if features_df is None or len(features_df) == 0:
                return {'probability_up': 0.5, 'confidence': 0.0}
            
            # Get latest features
            latest_features = features_df.iloc[-1].to_dict()
            
            # Get prediction
            prediction = self.ml_predictor.predict_probability(latest_features)
            return prediction
            
        except Exception as e:
            self.logger.error(f"ML prediction error for {symbol}: {e}")
            return {'probability_up': 0.5, 'confidence': 0.0}
    
    def _analyze_technicals(self, data: pd.DataFrame) -> Dict:
        """Analyze technical indicators"""
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9, adjust=False).mean()
        
        # Determine MACD signal
        if macd.iloc[-1] > signal_line.iloc[-1] and macd.iloc[-2] <= signal_line.iloc[-2]:
            macd_signal = 'bullish'
        elif macd.iloc[-1] < signal_line.iloc[-1] and macd.iloc[-2] >= signal_line.iloc[-2]:
            macd_signal = 'bearish'
        else:
            macd_signal = 'neutral'
        
        return {
            'rsi': rsi,
            'macd': macd.iloc[-1],
            'macd_signal': macd_signal
        }
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate annualized volatility"""
        returns = data['Close'].pct_change().dropna()
        return returns.std() * np.sqrt(252)
    
    def _calculate_risk_score(self, symbol: str, volatility: float) -> float:
        """Calculate risk score (0-1, higher is riskier)"""
        # Base risk on volatility
        vol_risk = min(volatility / 0.5, 1.0)  # Cap at 50% annual volatility
        
        # Add other risk factors
        # Could add sector concentration, correlation, etc.
        
        return vol_risk
    
    def execute_signals(self, signals: List[TradingSignal]) -> Dict:
        """Execute trading signals"""
        results = {
            'executed': [],
            'failed': [],
            'skipped': []
        }
        
        # Check risk limits before executing
        if not self._check_risk_limits():
            self.logger.warning("Risk limits exceeded, skipping trades")
            return results
        
        # Sort signals by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        for signal in signals:
            try:
                if signal.action == 'BUY':
                    order = self.alpaca_bot.place_buy_order(
                        symbol=signal.symbol,
                        quantity=signal.position_size,
                        reason='; '.join(signal.reasons)
                    )
                    if order:
                        results['executed'].append({
                            'symbol': signal.symbol,
                            'action': signal.action,
                            'quantity': signal.position_size,
                            'order_id': order.id
                        })
                    else:
                        results['failed'].append(signal.symbol)
                        
                elif signal.action == 'SELL':
                    success = self.alpaca_bot.close_position(
                        symbol=signal.symbol,
                        reason='; '.join(signal.reasons)
                    )
                    if success:
                        results['executed'].append({
                            'symbol': signal.symbol,
                            'action': signal.action
                        })
                    else:
                        results['failed'].append(signal.symbol)
                        
                else:  # HOLD
                    results['skipped'].append(signal.symbol)
                    
            except Exception as e:
                self.logger.error(f"Error executing signal for {signal.symbol}: {e}")
                results['failed'].append(signal.symbol)
        
        return results
    
    def _check_risk_limits(self) -> bool:
        """Check if portfolio meets risk limits"""
        try:
            metrics = self.alpaca_bot.get_portfolio_metrics()
            
            # Check exposure limit
            if metrics['cash_percentage'] < (1 - self.max_portfolio_exposure):
                self.logger.warning("Portfolio exposure limit reached")
                return False
            
            # Could add more risk checks here
            # - Sector concentration
            # - Correlation limits
            # - VaR limits
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False
    
    def rebalance_portfolio(self) -> Dict:
        """Rebalance portfolio to target weights"""
        results = {
            'rebalanced': [],
            'failed': []
        }
        
        try:
            # Get current positions
            positions = self.alpaca_bot.get_positions()
            if not positions:
                return results
            
            # Calculate current weights
            metrics = self.alpaca_bot.get_portfolio_metrics()
            total_value = metrics['total_value']
            
            current_weights = {}
            for symbol, pos_data in metrics['positions'].items():
                current_weights[symbol] = pos_data['market_value'] / total_value
            
            # Get target weights from optimizer
            symbols = list(positions.keys())
            historical_data = {}
            
            for symbol in symbols:
                data = self.data_collector.fetch_stock_data(symbol, period='1y')
                if data is not None:
                    historical_data[symbol] = data
            
            if historical_data:
                target_weights = self.portfolio_optimizer.optimize_portfolio(
                    list(historical_data.keys()),
                    historical_data
                )
                
                # Execute rebalancing trades
                for symbol, target_weight in target_weights.items():
                    current_weight = current_weights.get(symbol, 0)
                    weight_diff = abs(target_weight - current_weight)
                    
                    if weight_diff > self.rebalance_threshold:
                        # Calculate shares to trade
                        target_value = total_value * target_weight
                        current_value = total_value * current_weight
                        value_diff = target_value - current_value
                        
                        quote = self.alpaca_bot.get_latest_quote(symbol)
                        if quote:
                            price = quote['ask_price']
                            shares = int(abs(value_diff) / price)
                            
                            if value_diff > 0 and shares > 0:
                                # Buy more
                                order = self.alpaca_bot.place_buy_order(
                                    symbol, shares, reason="Portfolio rebalancing"
                                )
                                if order:
                                    results['rebalanced'].append(symbol)
                            elif value_diff < 0 and shares > 0:
                                # Sell some
                                order = self.alpaca_bot.place_sell_order(
                                    symbol, shares, reason="Portfolio rebalancing"
                                )
                                if order:
                                    results['rebalanced'].append(symbol)
                                    
        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio: {e}")
            
        return results
    
    def manage_risk(self) -> Dict:
        """Manage portfolio risk through stop-loss and take-profit"""
        results = {
            'stopped_out': [],
            'took_profits': []
        }
        
        positions = self.alpaca_bot.get_positions()
        
        for symbol in positions:
            # Check stop-loss
            should_stop, pnl = self.alpaca_bot.check_stop_loss(symbol)
            if should_stop:
                success = self.alpaca_bot.close_position(
                    symbol, 
                    reason=f"Stop-loss triggered at {pnl:.1%} loss"
                )
                if success:
                    results['stopped_out'].append(symbol)
            
            # Check take-profit
            should_take, pnl = self.alpaca_bot.check_take_profit(symbol)
            if should_take:
                success = self.alpaca_bot.close_position(
                    symbol,
                    reason=f"Take-profit triggered at {pnl:.1%} gain"
                )
                if success:
                    results['took_profits'].append(symbol)
        
        return results
    
    def generate_portfolio_report(self) -> Dict:
        """Generate comprehensive portfolio report"""
        metrics = self.alpaca_bot.get_portfolio_metrics()
        
        # Calculate additional metrics
        sharpe_ratio = self._calculate_sharpe_ratio()
        max_drawdown = self._calculate_max_drawdown()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': metrics['total_value'],
            'cash': metrics['cash'],
            'positions_value': metrics['positions_value'],
            'num_positions': metrics['num_positions'],
            'total_return': (metrics['total_value'] - self.initial_capital) / self.initial_capital,
            'unrealized_pnl': metrics['total_unrealized_pnl'],
            'unrealized_pnl_pct': metrics['total_unrealized_pnl_pct'],
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'positions': metrics['positions'],
            'risk_metrics': {
                'portfolio_volatility': self._calculate_portfolio_volatility(),
                'portfolio_beta': self._calculate_portfolio_beta(),
                'var_95': self._calculate_var(0.95)
            }
        }
        
        return report
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from trade history"""
        # Simplified calculation - would need full returns history
        try:
            with open('trades_log.json', 'r') as f:
                trades = json.load(f)
            
            if len(trades) < 2:
                return 0.0
            
            # Calculate returns between trades
            returns = []
            for i in range(1, len(trades)):
                prev_value = trades[i-1]['account_value']
                curr_value = trades[i]['account_value']
                if prev_value > 0:
                    ret = (curr_value - prev_value) / prev_value
                    returns.append(ret)
            
            if returns:
                return np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
            
        except Exception:
            pass
            
        return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        # Would need full equity curve - simplified for now
        return 0.0
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility"""
        # Simplified - would need correlation matrix
        return 0.15  # Placeholder
    
    def _calculate_portfolio_beta(self) -> float:
        """Calculate portfolio beta vs market"""
        # Simplified - would need market data
        return 1.0  # Placeholder
    
    def _calculate_var(self, confidence: float) -> float:
        """Calculate Value at Risk"""
        # Simplified - would need full returns distribution
        return 0.02  # Placeholder 2% VaR