import os
import json
from typing import Dict, List, Any
from data_collection import DataCollectionAgent
from technical_analysis import technical_analyst_agent
from risk_manager import risk_manager_agent
from news_analyst import news_analyst
from portfolio_manager_rule_based import PortfolioManagerAgent

class MarketCoordinatorAgent:
    """
    Rule-based master agent that coordinates all other agents and provides 
    high-level market analysis and investment strategies (no LLM).
    """
    def __init__(self):
        self.data_collector = DataCollectionAgent()
        self.tech_analyst = technical_analyst_agent()
        self.risk_manager = risk_manager_agent()
        self.news_analyst = news_analyst()
        self.portfolio_manager = PortfolioManagerAgent()
    
    def comprehensive_market_analysis(self, tickers: List[str], portfolio_value: float = 100000) -> Dict[str, Any]:
        print(f"🎯 Market Coordinator (Rule-Based) analyzing: {', '.join(tickers)}")
        market_data = {}
        for ticker in tickers:
            print(f"\n📈 Analyzing {ticker}...")
            portfolio_analysis = self.portfolio_manager.analyze_stock(ticker, portfolio_value)
            market_context = self._get_market_context(ticker)
            market_data[ticker] = {
                "portfolio_analysis": portfolio_analysis,
                "market_context": market_context
            }
        coordination_analysis = self._rule_based_coordination(market_data, portfolio_value)
        return {
            "coordinator_analysis": coordination_analysis,
            "individual_stocks": market_data,
            "portfolio_value": portfolio_value
        }
    def _get_market_context(self, ticker: str) -> Dict[str, Any]:
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("forwardPE", None),
                "dividend_yield": info.get("dividendYield", None),
                "52_week_high": info.get("fiftyTwoWeekHigh", None),
                "52_week_low": info.get("fiftyTwoWeekLow", None),
            }
        except Exception as e:
            return {"error": f"Could not fetch market context: {e}"}
    def _rule_based_coordination(self, market_data: Dict[str, Any], portfolio_value: float) -> str:
        analysis = []
        analysis.append(f"📊 MARKET COORDINATOR ANALYSIS (Rule-Based)")
        analysis.append(f"Portfolio Value: ${portfolio_value:,}\n")
        buy_stocks = []
        hold_stocks = []
        sell_stocks = []
        for ticker, data in market_data.items():
            rec = data["portfolio_analysis"]["recommendation"]
            confidence = data["portfolio_analysis"]["confidence"]
            risk = data["portfolio_analysis"]["risk_assessment"]
            stock_summary = f"{ticker} ({rec}, {confidence:.1%} confidence, {risk} risk)"
            if rec == "BUY":
                buy_stocks.append(stock_summary)
            elif rec == "SELL":
                sell_stocks.append(stock_summary)
            else:
                hold_stocks.append(stock_summary)
        if buy_stocks:
            analysis.append("🟢 BUY RECOMMENDATIONS:")
            for stock in buy_stocks:
                analysis.append(f"  • {stock}")
        if hold_stocks:
            analysis.append("\n🟡 HOLD RECOMMENDATIONS:")
            for stock in hold_stocks:
                analysis.append(f"  • {stock}")
        if sell_stocks:
            analysis.append("\n🔴 SELL RECOMMENDATIONS:")
            for stock in sell_stocks:
                analysis.append(f"  • {stock}")
        analysis.append(f"\n⚠️ RISK ANALYSIS:")
        total_stocks = len(market_data)
        high_risk = sum(1 for data in market_data.values() if "High" in data["portfolio_analysis"]["risk_assessment"])
        if high_risk > total_stocks * 0.5:
            analysis.append("  • Portfolio has high concentration of risky assets")
            analysis.append("  • Consider diversification or position size reduction")
        else:
            analysis.append("  • Risk levels appear balanced across portfolio")
        analysis.append(f"\n📋 RECOMMENDED ACTIONS:")
        if buy_stocks:
            analysis.append(f"  1. Consider positions in {len(buy_stocks)} recommended BUY stocks")
        if sell_stocks:
            analysis.append(f"  2. Review and consider exiting {len(sell_stocks)} SELL recommendations")
        analysis.append(f"  3. Monitor market conditions and news for all positions")
        analysis.append(f"  4. Rebalance portfolio based on risk tolerance")
        return "\n".join(analysis)
    def market_alert_system(self, tickers: List[str]) -> Dict[str, Any]:
        alerts = []
        for ticker in tickers:
            try:
                risk_data = self.risk_manager.calculate_risk_score(ticker)
                news_data = self.news_analyst.analyze_stock_news(ticker)
                if risk_data['risk_score'] > 8:
                    alerts.append({
                        "ticker": ticker,
                        "type": "HIGH_RISK",
                        "message": f"{ticker} shows very high risk (score: {risk_data['risk_score']:.1f})",
                        "priority": "HIGH"
                    })
                if news_data.get('sentiment') == 'very_negative':
                    alerts.append({
                        "ticker": ticker,
                        "type": "NEGATIVE_NEWS",
                        "message": f"{ticker} has very negative news sentiment",
                        "priority": "MEDIUM"
                    })
            except Exception as e:
                alerts.append({
                    "ticker": ticker,
                    "type": "ERROR",
                    "message": f"Error analyzing {ticker}: {e}",
                    "priority": "LOW"
                })
        return {"alerts": alerts, "alert_count": len(alerts)}