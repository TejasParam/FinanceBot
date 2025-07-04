import os
import json
from typing import Dict, List, Any
from data_collection import DataCollectionAgent
from technical_analysis import technical_analyst_agent
from risk_manager import risk_manager_agent
from news_analyst import news_analyst

class PortfolioManagerAgent:
    """
    Rule-based portfolio manager that makes high-level investment decisions
    by reasoning about technical analysis, risk metrics, and news sentiment.
    """
    
    def __init__(self):
        # Initialize other agents as tools
        self.data_collector = DataCollectionAgent()
        self.tech_analyst = technical_analyst_agent()
        self.risk_manager = risk_manager_agent()
        self.news_analyst = news_analyst()
    
    def analyze_stock(self, ticker: str, portfolio_value: float = 100000) -> Dict[str, Any]:
        """
        Comprehensive stock analysis using only rule-based logic over all agent data
        """
        try:
            print(f"🤖 Portfolio Manager (Rule-Based) analyzing {ticker}...")
            analysis_data = self._gather_analysis_data(ticker, portfolio_value)
            result = self._rule_based_analysis(ticker, analysis_data)
            return result
        except Exception as e:
            print(f"Error in portfolio analysis: {e}")
            return {
                "recommendation": "HOLD",
                "confidence": 0.5,
                "reasoning": "Error occurred during analysis",
                "risk_assessment": "Unknown"
            }
    
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
            data['news_sentiment'] = news_data.get('sentiment', 'neutral')
            data['news_summary'] = news_data.get('summary', 'No recent news')
            data['market_impact'] = news_data.get('market_impact', 'low')
        except Exception as e:
            print(f"    ⚠️ News analysis error: {e}")
            data['news_error'] = str(e)
        return data
    
    def _rule_based_analysis(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple rule-based logic for recommendation
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