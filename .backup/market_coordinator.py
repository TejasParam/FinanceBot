import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, List, Any
from data_collection import DataCollectionAgent
from technical_analysis import technical_analyst_agent
from risk_manager import risk_manager_agent
from news_analyst import news_analyst
from portfolio_manager import PortfolioManagerAgent

class MarketCoordinatorAgent:
    """
    Master LLM agent that coordinates all other agents and provides 
    high-level market analysis and investment strategies.
    """
    
    def __init__(self):
        # Initialize all specialized agents
        self.data_collector = DataCollectionAgent()
        self.tech_analyst = technical_analyst_agent()
        self.risk_manager = risk_manager_agent()
        self.news_analyst = news_analyst()
        self.portfolio_manager = PortfolioManagerAgent()
        
        # Load Llama 2 model and tokenizer from local path (no login required)
        self.llama_tokenizer = AutoTokenizer.from_pretrained("/Users/tanis/llama-2-7b-chat-hf")
        self.llama_model = AutoModelForCausalLM.from_pretrained("/Users/tanis/llama-2-7b-chat-hf")
        self.llm_available = True
    
    def comprehensive_market_analysis(self, tickers: List[str], portfolio_value: float = 100000) -> Dict[str, Any]:
        """
        Comprehensive market analysis coordinating all agents
        """
        print(f"🎯 Market Coordinator analyzing: {', '.join(tickers)}")
        
        # Gather comprehensive data
        market_data = {}
        
        for ticker in tickers:
            print(f"\n📈 Analyzing {ticker}...")
            
            # Get portfolio manager analysis (which uses all other agents)
            portfolio_analysis = self.portfolio_manager.analyze_stock(ticker, portfolio_value)
            
            # Get additional market context
            market_context = self._get_market_context(ticker)
            
            market_data[ticker] = {
                "portfolio_analysis": portfolio_analysis,
                "market_context": market_context
            }
        
        # Generate coordinated analysis
        if self.llm_available:
            coordination_analysis = self._llm_coordination_analysis(market_data, portfolio_value)
        else:
            coordination_analysis = self._rule_based_coordination(market_data, portfolio_value)
        
        return {
            "coordinator_analysis": coordination_analysis,
            "individual_stocks": market_data,
            "portfolio_value": portfolio_value
        }
    
    def _get_market_context(self, ticker: str) -> Dict[str, Any]:
        """Get additional market context for coordination"""
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
    
    def _llm_coordination_analysis(self, market_data: Dict[str, Any], portfolio_value: float) -> str:
        """Use Mistral for sophisticated market coordination (replaces Llama 2)"""
        
        coordination_prompt = f"""
You are a senior portfolio manager overseeing a comprehensive market analysis system. 
You have access to detailed analysis from specialized agents for multiple stocks.

PORTFOLIO VALUE: ${portfolio_value:,}

STOCK ANALYSES:
{json.dumps(market_data, indent=2, default=str)}

As the coordinating agent, provide:

1. MARKET OVERVIEW
   - Current market sentiment and conditions
   - Sector analysis and trends
   - Key risks and opportunities

2. PORTFOLIO STRATEGY
   - Optimal portfolio allocation across analyzed stocks
   - Risk diversification recommendations
   - Timeline and entry/exit strategies

3. PRIORITIZED ACTIONS
   - Immediate actions to take
   - Stocks to monitor closely
   - Risk management adjustments

4. MARKET TIMING
   - Best entry points based on technical and fundamental analysis
   - Market cycle considerations
   - Economic factors to watch

5. ALTERNATIVE CONSIDERATIONS
   - Stocks to research further
   - Sectors to explore
   - Hedging strategies

Provide actionable insights that synthesize all agent analyses into a coherent investment strategy.
Consider correlation between stocks, market conditions, and portfolio optimization.
"""
        
        try:
            input_ids = self.llama_tokenizer(coordination_prompt, return_tensors="pt").input_ids
            with torch.no_grad():
                output = self.llama_model.generate(input_ids, max_new_tokens=512)
            response_text = self.llama_tokenizer.decode(output[0], skip_special_tokens=True)
            return response_text
        except Exception as e:
            return f"LLM coordination error: {e}\nFalling back to rule-based analysis..."
    
    def _rule_based_coordination(self, market_data: Dict[str, Any], portfolio_value: float) -> str:
        """Rule-based coordination when LLM is not available"""
        
        analysis = []
        analysis.append(f"📊 MARKET COORDINATOR ANALYSIS (Rule-Based)")
        analysis.append(f"Portfolio Value: ${portfolio_value:,}\n")
        
        # Analyze recommendations
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
        
        # Portfolio recommendations
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
        
        # Risk assessment
        analysis.append(f"\n⚠️ RISK ANALYSIS:")
        total_stocks = len(market_data)
        high_risk = sum(1 for data in market_data.values() 
                       if "High" in data["portfolio_analysis"]["risk_assessment"])
        
        if high_risk > total_stocks * 0.5:
            analysis.append("  • Portfolio has high concentration of risky assets")
            analysis.append("  • Consider diversification or position size reduction")
        else:
            analysis.append("  • Risk levels appear balanced across portfolio")
        
        # Action items
        analysis.append(f"\n📋 RECOMMENDED ACTIONS:")
        if buy_stocks:
            analysis.append(f"  1. Consider positions in {len(buy_stocks)} recommended BUY stocks")
        if sell_stocks:
            analysis.append(f"  2. Review and consider exiting {len(sell_stocks)} SELL recommendations")
        analysis.append(f"  3. Monitor market conditions and news for all positions")
        analysis.append(f"  4. Rebalance portfolio based on risk tolerance")
        
        return "\n".join(analysis)
    
    def market_alert_system(self, tickers: List[str]) -> Dict[str, Any]:
        """Generate market alerts based on agent analyses"""
        alerts = []
        
        for ticker in tickers:
            try:
                # Quick analysis for alerts
                risk_data = self.risk_manager.calculate_risk_score(ticker)
                news_data = self.news_analyst.analyze_stock_news(ticker)
                
                # Generate alerts based on conditions
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
                
                # Add more alert conditions as needed
                
            except Exception as e:
                alerts.append({
                    "ticker": ticker,
                    "type": "ERROR",
                    "message": f"Error analyzing {ticker}: {e}",
                    "priority": "LOW"
                })
        
        return {"alerts": alerts, "alert_count": len(alerts)}

# Example usage
if __name__ == "__main__":
    coordinator = MarketCoordinatorAgent()
    
    # Comprehensive analysis
    tickers = ["AAPL", "GOOGL", "MSFT"]
    analysis = coordinator.comprehensive_market_analysis(tickers, 100000)
    
    print("🎯 MARKET COORDINATOR ANALYSIS:")
    print("=" * 50)
    print(analysis["coordinator_analysis"])
    
    # Market alerts
    alerts = coordinator.market_alert_system(tickers)
    if alerts["alert_count"] > 0:
        print(f"\n🚨 MARKET ALERTS ({alerts['alert_count']}):")
        for alert in alerts["alerts"]:
            print(f"  [{alert['priority']}] {alert['message']}")
