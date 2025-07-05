import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from data_collection import DataCollectionAgent
from technical_analysis import technical_analyst_agent
from risk_manager import risk_manager_agent
from news_analyst import news_analyst
try:
    from ml_predictor import MLPredictor
    from backtest_engine import BacktestEngine
    ML_AVAILABLE = True
except ImportError:
    print("⚠️ ML components not available. Install scikit-learn and matplotlib for full functionality.")
    ML_AVAILABLE = False

class AdvancedPortfolioManagerAgent:
    """
    Advanced portfolio manager with machine learning, backtesting, and enhanced analysis.
    Provides both rule-based and ML-driven investment decisions with comprehensive evaluation.
    """
    
    def __init__(self, use_ml: bool = True):
        # Initialize other agents as tools
        self.data_collector = DataCollectionAgent()
        self.tech_analyst = technical_analyst_agent()
        self.risk_manager = risk_manager_agent()
        self.news_analyst = news_analyst()
        
        # Initialize ML and backtesting components
        self.use_ml = use_ml and ML_AVAILABLE
        if self.use_ml:
            self.ml_predictor = MLPredictor()
            self.backtest_engine = BacktestEngine()
            self.models_path = "models/portfolio_models.pkl"
            self._load_or_train_models()
        
        # Enhanced rule weights (can be optimized through backtesting)
        self.rule_weights = {
            'technical': 0.4,
            'sentiment': 0.2,
            'risk': 0.3,
            'ml_prediction': 0.1 if self.use_ml else 0.0
        }
    
    def _load_or_train_models(self):
        """Load existing models or indicate training is needed"""
        if not os.path.exists("models"):
            os.makedirs("models")
        
        if not self.ml_predictor.load_models(self.models_path):
            print("📚 ML models not found. Use train_ml_models() to train on historical data.")
    
    def train_ml_models(self, ticker: str, period: str = "2y") -> Dict[str, Any]:
        """
        Train ML models on historical data for a specific ticker
        """
        if not self.use_ml:
            return {"error": "ML components not available"}
        
        print(f"🎓 Training ML models for {ticker}...")
        
        # Get extended historical data for training
        try:
            historical_data = self.data_collector.get_historical_data(ticker, period=period)
            if historical_data is None or len(historical_data) < 200:
                return {"error": "Insufficient historical data for training"}
            
            # Train the models
            training_results = self.ml_predictor.train_models(historical_data)
            
            if "error" not in training_results:
                # Save trained models
                self.ml_predictor.save_models(self.models_path)
                print("✅ ML models trained and saved successfully!")
                
                return {
                    "status": "success",
                    "training_results": training_results,
                    "data_points": len(historical_data),
                    "models_saved": self.models_path
                }
            else:
                return training_results
                
        except Exception as e:
            return {"error": f"Training failed: {str(e)}"}
    
    def backtest_strategy(self, ticker: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Backtest the current strategy on historical data
        """
        if not self.use_ml:
            return {"error": "Backtesting requires ML components"}
        
        print(f"📈 Backtesting strategy for {ticker}...")
        
        try:
            # Get historical data
            period = "2y" if not start_date else "max"
            historical_data = self.data_collector.get_historical_data(ticker, period=period)
            
            if historical_data is None or len(historical_data) < 50:
                return {"error": "Insufficient data for backtesting"}
            
            # Create a strategy function that uses our analysis
            def strategy_function(data):
                if len(data) < 20:  # Need minimum data for analysis
                    return {"recommendation": "HOLD", "confidence": 0.5}
                
                # Simulate our analysis on historical data
                latest_data = data.tail(1)
                ticker_temp = ticker  # Capture ticker for the function
                
                # This is a simplified version for backtesting
                # In practice, you'd want to recreate the full analysis context
                fake_analysis_data = {
                    'current_price': float(latest_data['Close'].iloc[0]),
                    'rsi': self._calculate_simple_rsi(data['Close'].tail(14)),
                    'news_sentiment': 'neutral',  # Simplified for backtesting
                    'risk_category': 'Medium'  # Simplified for backtesting
                }
                
                return self._rule_based_analysis(ticker_temp, fake_analysis_data)
            
            # Run backtest
            backtest_results = self.backtest_engine.backtest_strategy(
                historical_data, strategy_function, start_date=start_date, end_date=end_date
            )
            
            return backtest_results
            
        except Exception as e:
            return {"error": f"Backtesting failed: {str(e)}"}
    
    def _calculate_simple_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Simple RSI calculation for backtesting"""
        if len(prices) < period:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
        return 100 - (100 / (1 + rs)) if rs != 0 else 50.0
    
    def analyze_stock(self, ticker: str, portfolio_value: float = 100000) -> Dict[str, Any]:
        """
        Comprehensive stock analysis using rule-based logic enhanced with ML predictions
        """
        try:
            print(f"🤖 Advanced Portfolio Manager analyzing {ticker}...")
            analysis_data = self._gather_analysis_data(ticker, portfolio_value)
            
            # Get ML prediction if available
            if self.use_ml and self.ml_predictor.is_trained:
                ml_features = self._extract_ml_features(analysis_data)
                ml_prediction = self.ml_predictor.predict_probability(ml_features)
                analysis_data['ml_prediction'] = ml_prediction
            
            result = self._enhanced_analysis(ticker, analysis_data)
            return result
        except Exception as e:
            print(f"Error in portfolio analysis: {e}")
            return {
                "recommendation": "HOLD",
                "confidence": 0.5,
                "reasoning": "Error occurred during analysis",
                "risk_assessment": "Unknown"
            }
    
    def _extract_ml_features(self, analysis_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for ML prediction from analysis data"""
        features = {}
        
        # Technical features
        features['rsi'] = analysis_data.get('rsi', 50.0)
        
        # MACD features
        macd_data = analysis_data.get('macd', {})
        if isinstance(macd_data, dict):
            features['macd'] = macd_data.get('macd', 0.0)
            features['macd_signal'] = macd_data.get('signal', 0.0)
        else:
            features['macd'] = 0.0
            features['macd_signal'] = 0.0
        
        # Moving averages
        ma_data = analysis_data.get('moving_averages', {})
        features['sma_20'] = ma_data.get('sma_20', analysis_data.get('current_price', 100))
        features['ema_12'] = ma_data.get('ema_12', analysis_data.get('current_price', 100))
        
        # Risk features
        features['beta'] = analysis_data.get('beta', 1.0)
        features['volatility'] = analysis_data.get('volatility', 0.2)
        features['sharpe_ratio'] = analysis_data.get('sharpe_ratio', 0.0)
        
        # Price-based features
        current_price = analysis_data.get('current_price', 100)
        features['price_vs_sma20'] = current_price / features['sma_20'] if features['sma_20'] > 0 else 1.0
        
        # Bollinger Bands
        bb_data = analysis_data.get('bollinger_bands', {})
        if isinstance(bb_data, dict):
            bb_upper = bb_data.get('upper', current_price * 1.02)
            bb_lower = bb_data.get('lower', current_price * 0.98)
            features['bb_position'] = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        else:
            features['bb_position'] = 0.5
        
        return features
    
    def _gather_analysis_data(self, ticker: str, portfolio_value: float) -> Dict[str, Any]:
        # ...existing code from original _gather_analysis_data...
        data = {}
        print("  📊 Gathering technical analysis...")
        try:
            tech_data = self.tech_analyst.getData(ticker)
            if tech_data and len(tech_data) > 0:
                latest_data = tech_data[0].tail(1)
                data['current_price'] = float(latest_data['Close'].iloc[0])
                data['rsi'] = self.tech_analyst.calculate_rsi(ticker)
                data['macd'] = self.tech_analyst.calculate_macd(ticker)
                data['bollinger_bands'] = self.tech_analyst.calculate_bollinger_bands(ticker)
                data['moving_averages'] = {
                    'sma_20': self.tech_analyst.calculate_sma(ticker, 20),
                    'ema_12': self.tech_analyst.calculate_ema(ticker, 12)
                }
        except Exception as e:
            print(f"    ⚠️ Technical analysis error: {e}")
            data['technical_error'] = str(e)
        print("  ⚠️ Gathering risk analysis...")
        try:
            risk_data = self.risk_manager.calculate_risk_score(ticker)
            data['risk_score'] = risk_data['risk_score']
            data['risk_category'] = risk_data['risk_category']
            data['beta'] = self.risk_manager.calculate_beta(ticker)
            data['volatility'] = self.risk_manager.calculate_volatility(ticker)
            data['sharpe_ratio'] = self.risk_manager.calculate_sharpe_ratio(ticker)
            data['position_sizing'] = self.risk_manager.recommend_position_size(ticker, portfolio_value)
        except Exception as e:
            print(f"    ⚠️ Risk analysis error: {e}")
            data['risk_error'] = str(e)
        print("  📰 Gathering news sentiment...")
        try:
            news_data = self.news_analyst.analyze_stock_news(ticker)
            if isinstance(news_data, list) and len(news_data) > 0:
                # Calculate overall sentiment from news items
                sentiments = [item.get('sentiment', 'neutral') for item in news_data]
                positive_count = sentiments.count('positive')
                negative_count = sentiments.count('negative')
                if positive_count > negative_count:
                    overall_sentiment = 'positive'
                elif negative_count > positive_count:
                    overall_sentiment = 'negative' 
                else:
                    overall_sentiment = 'neutral'
                
                data['news_sentiment'] = overall_sentiment
                data['news_summary'] = f"Analyzed {len(news_data)} news items"
                data['market_impact'] = 'high' if len(news_data) > 2 else 'medium'
            else:
                data['news_sentiment'] = 'neutral'
                data['news_summary'] = 'No recent news'
                data['market_impact'] = 'low'
        except Exception as e:
            print(f"    ⚠️ News analysis error: {e}")
            data['news_error'] = str(e)
        return data
    
    def _enhanced_analysis(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced rule-based analysis that incorporates ML predictions and weighted scoring
        """
        # Get base rule-based analysis
        base_analysis = self._rule_based_analysis(ticker, data)
        
        # Calculate component scores
        technical_score = self._calculate_technical_score(data)
        sentiment_score = self._calculate_sentiment_score(data)
        risk_score = self._calculate_risk_score(data)
        
        # ML prediction score
        ml_score = 0.5  # Neutral default
        ml_confidence = 0.0
        if self.use_ml and 'ml_prediction' in data and 'error' not in data['ml_prediction']:
            ml_pred = data['ml_prediction']
            ml_score = ml_pred.get('probability_up', 0.5)
            ml_confidence = ml_pred.get('confidence', 0.0)
        
        # Calculate weighted composite score
        composite_score = (
            technical_score * self.rule_weights['technical'] +
            sentiment_score * self.rule_weights['sentiment'] +
            risk_score * self.rule_weights['risk'] +
            ml_score * self.rule_weights['ml_prediction']
        )
        
        # Determine recommendation based on composite score
        if composite_score > 0.65:
            recommendation = "BUY"
            confidence = min(0.9, 0.5 + (composite_score - 0.65) * 2)
        elif composite_score < 0.35:
            recommendation = "SELL"
            confidence = min(0.9, 0.5 + (0.35 - composite_score) * 2)
        else:
            recommendation = "HOLD"
            confidence = 0.6
        
        # Enhanced reasoning
        reasoning_parts = [base_analysis['reasoning']]
        
        if self.use_ml and ml_confidence > 0.6:
            ml_direction = "upward" if ml_score > 0.5 else "downward"
            reasoning_parts.append(f"ML model predicts {ml_direction} movement with {ml_confidence:.1%} confidence.")
        
        reasoning_parts.append(f"Composite score: {composite_score:.2f} (Technical: {technical_score:.2f}, Sentiment: {sentiment_score:.2f}, Risk: {risk_score:.2f})")
        
        # Enhanced result
        enhanced_result = base_analysis.copy()
        enhanced_result.update({
            "recommendation": recommendation,
            "confidence": confidence,
            "reasoning": " ".join(reasoning_parts),
            "composite_score": composite_score,
            "component_scores": {
                "technical": technical_score,
                "sentiment": sentiment_score,
                "risk": risk_score,
                "ml_prediction": ml_score
            },
            "ml_analysis": data.get('ml_prediction', {}) if self.use_ml else None
        })
        
        return enhanced_result
    
    def _rule_based_analysis(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Original simple rule-based logic for recommendation (kept for compatibility)
        """
        # Example rules (customize as needed)
        rsi = data.get('rsi', 50)
        sentiment = data.get('news_sentiment', 'neutral')
        risk = data.get('risk_category', 'Unknown')
        recommendation = "HOLD"
        confidence = 0.5
        reasoning = []
        
        if rsi < 30 and sentiment in ('positive', 'neutral') and risk != 'High':
            recommendation = "BUY"
            confidence = 0.8
            reasoning.append("RSI indicates oversold, sentiment not negative, risk not high.")
        elif rsi > 70 or sentiment == 'very_negative' or risk == 'High':
            recommendation = "SELL"
            confidence = 0.8
            reasoning.append("RSI indicates overbought, or negative sentiment, or high risk.")
        else:
            recommendation = "HOLD"
            confidence = 0.6
            reasoning.append("No strong buy/sell signals.")
            
        return {
            "ticker": ticker,
            "recommendation": recommendation,
            "confidence": confidence,
            "reasoning": " ".join(reasoning),
            "risk_assessment": risk,
            "technical_data": {
                "rsi": rsi,
                "beta": data.get('beta'),
                "volatility": data.get('volatility')
            },
            "position_sizing": data.get('position_sizing'),
            "sentiment": sentiment
        }
    
    def optimize_strategy(self, ticker: str, optimization_period: str = "1y") -> Dict[str, Any]:
        """
        Optimize rule weights using historical performance
        """
        if not self.use_ml:
            return {"error": "Optimization requires ML components"}
        
        print(f"🔧 Optimizing strategy for {ticker}...")
        
        try:
            # Test different weight combinations
            weight_combinations = [
                {'technical': 0.5, 'sentiment': 0.2, 'risk': 0.3, 'ml_prediction': 0.0},
                {'technical': 0.4, 'sentiment': 0.2, 'risk': 0.3, 'ml_prediction': 0.1},
                {'technical': 0.3, 'sentiment': 0.3, 'risk': 0.3, 'ml_prediction': 0.1},
                {'technical': 0.4, 'sentiment': 0.1, 'risk': 0.4, 'ml_prediction': 0.1},
                {'technical': 0.6, 'sentiment': 0.1, 'risk': 0.2, 'ml_prediction': 0.1},
            ]
            
            best_weights = None
            best_return = -float('inf')
            results = []
            
            for weights in weight_combinations:
                # Temporarily set weights
                original_weights = self.rule_weights.copy()
                self.rule_weights = weights
                
                # Backtest with these weights
                backtest_result = self.backtest_strategy(ticker)
                
                if 'error' not in backtest_result:
                    total_return = backtest_result['total_return']
                    sharpe = backtest_result['metrics'].get('sharpe_ratio', 0)
                    
                    # Score = return * sharpe ratio (risk-adjusted return)
                    score = total_return * max(0, sharpe)
                    
                    results.append({
                        'weights': weights.copy(),
                        'return': total_return,
                        'sharpe': sharpe,
                        'score': score
                    })
                    
                    if score > best_return:
                        best_return = score
                        best_weights = weights.copy()
                
                # Restore original weights
                self.rule_weights = original_weights
            
            if best_weights:
                self.rule_weights = best_weights
                print(f"✅ Optimized weights: {best_weights}")
                return {
                    "status": "success",
                    "best_weights": best_weights,
                    "best_score": best_return,
                    "all_results": results
                }
            else:
                return {"error": "Optimization failed - no valid backtests"}
                
        except Exception as e:
            return {"error": f"Optimization failed: {str(e)}"}
    
    def get_market_regime_analysis(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze current market regime (trending, ranging, volatile)
        """
        try:
            print(f"📊 Analyzing market regime for {ticker}...")
            
            # Get recent data
            data = self.data_collector.get_historical_data(ticker, period="3mo")
            if data is None or len(data) < 50:
                return {"error": "Insufficient data for regime analysis"}
            
            # Calculate various metrics
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Trend strength (using linear regression slope)
            x = np.arange(len(data))
            slope = np.polyfit(x, data['Close'], 1)[0]
            trend_strength = abs(slope) / data['Close'].mean()
            
            # Range-bound detection (price within channels)
            price_range = (data['Close'].max() - data['Close'].min()) / data['Close'].mean()
            
            # Regime classification
            if trend_strength > 0.001 and volatility < 0.3:
                regime = "Trending"
                confidence = min(0.9, trend_strength * 500)
            elif price_range < 0.15 and volatility < 0.25:
                regime = "Range-bound"
                confidence = min(0.9, (0.15 - price_range) * 6)
            elif volatility > 0.4:
                regime = "High Volatility"
                confidence = min(0.9, (volatility - 0.4) * 2)
            else:
                regime = "Transitional"
                confidence = 0.5
            
            return {
                "regime": regime,
                "confidence": confidence,
                "metrics": {
                    "volatility": volatility,
                    "trend_strength": trend_strength,
                    "price_range": price_range
                },
                "recommendation": self._regime_based_recommendation(regime, confidence)
            }
            
        except Exception as e:
            return {"error": f"Regime analysis failed: {str(e)}"}
    
    def _regime_based_recommendation(self, regime: str, confidence: float) -> str:
        """Provide strategy recommendation based on market regime"""
        if regime == "Trending" and confidence > 0.7:
            return "Use momentum strategies and trend-following indicators"
        elif regime == "Range-bound" and confidence > 0.7:
            return "Use mean-reversion strategies and support/resistance levels"
        elif regime == "High Volatility":
            return "Reduce position sizes and focus on risk management"
        else:
            return "Use balanced approach with multiple strategies"
    
    def _calculate_technical_score(self, data: Dict[str, Any]) -> float:
        """Calculate technical analysis score (0-1 scale)"""
        score = 0.5  # Neutral baseline
        
        # RSI scoring
        rsi = data.get('rsi', 50)
        if rsi < 30:  # Oversold - bullish
            score += 0.2
        elif rsi > 70:  # Overbought - bearish
            score -= 0.2
        
        # Moving average scoring
        current_price = data.get('current_price', 100)
        ma_data = data.get('moving_averages', {})
        sma_20 = ma_data.get('sma_20')
        if sma_20 and current_price > sma_20:
            score += 0.1  # Above SMA20 is bullish
        elif sma_20 and current_price < sma_20:
            score -= 0.1
        
        # MACD scoring
        macd_data = data.get('macd', {})
        if isinstance(macd_data, dict):
            macd = macd_data.get('macd', 0)
            signal = macd_data.get('signal', 0)
            if macd > signal:
                score += 0.1  # MACD above signal is bullish
            else:
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_sentiment_score(self, data: Dict[str, Any]) -> float:
        """Calculate sentiment score (0-1 scale)"""
        sentiment = data.get('news_sentiment', 'neutral')
        sentiment_scores = {
            'very_positive': 0.9,
            'positive': 0.7,
            'neutral': 0.5,
            'negative': 0.3,
            'very_negative': 0.1
        }
        return sentiment_scores.get(sentiment, 0.5)
    
    def _calculate_risk_score(self, data: Dict[str, Any]) -> float:
        """Calculate risk score (0-1 scale, where 1 is low risk)"""
        risk_category = data.get('risk_category', 'Medium')
        risk_scores = {
            'Low': 0.8,
            'Medium': 0.5,
            'High': 0.2,
            'Very High': 0.1
        }
        return risk_scores.get(risk_category, 0.5)
    
# For backward compatibility
PortfolioManagerAgent = AdvancedPortfolioManagerAgent