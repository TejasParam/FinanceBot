import os
import json
import google.generativeai as genai
from typing import Dict, List, Any
from data_collection import DataCollectionAgent
from technical_analysis import technical_analyst_agent
from risk_manager import risk_manager_agent
from news_analyst import news_analyst

class PortfolioManagerAgent:
    """
    LLM-powered portfolio manager that makes high-level investment decisions
    by reasoning about technical analysis, risk metrics, and news sentiment.
    """
    
    def __init__(self):
        # Initialize other agents as tools
        self.data_collector = DataCollectionAgent()
        self.tech_analyst = technical_analyst_agent()
        self.risk_manager = risk_manager_agent()
        self.news_analyst = news_analyst()
        
        # Configure Gemini
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-pro')
        
    def analyze_stock(self, ticker: str, portfolio_value: float = 100000) -> Dict[str, Any]:
        """
        Comprehensive stock analysis using LLM reasoning over all agent data
        """
        try:
            print(f"🤖 Portfolio Manager analyzing {ticker}...")
            
            # Gather data from all agents
            analysis_data = self._gather_analysis_data(ticker, portfolio_value)
            
            # Create comprehensive prompt for LLM analysis
            prompt = self._create_analysis_prompt(ticker, analysis_data)
            
            # Get LLM analysis
            response = self.model.generate_content(prompt)
            
            # Parse and structure the response
            result = self._parse_llm_response(response.text, analysis_data)
            
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
        """Gather data from all specialized agents"""
        data = {}
        
        # Technical Analysis
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
        
        # Risk Analysis
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
        
        # News Analysis
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
    
    def _create_analysis_prompt(self, ticker: str, data: Dict[str, Any]) -> str:
        """Create a comprehensive analysis prompt for the LLM"""
        
        prompt = f"""
You are an expert portfolio manager analyzing stock {ticker}. Based on the comprehensive data below, provide a detailed investment recommendation.

TECHNICAL ANALYSIS:
- Current Price: ${data.get('current_price', 'N/A')}
- RSI: {data.get('rsi', 'N/A')} (70+ overbought, 30- oversold)
- MACD: {data.get('macd', 'N/A')}
- Bollinger Bands: {data.get('bollinger_bands', 'N/A')}
- Moving Averages: {data.get('moving_averages', 'N/A')}

RISK ANALYSIS:
- Risk Score: {data.get('risk_score', 'N/A')}/10 ({data.get('risk_category', 'N/A')})
- Beta: {data.get('beta', 'N/A')} (market correlation)
- Volatility: {data.get('volatility', 'N/A')}
- Sharpe Ratio: {data.get('sharpe_ratio', 'N/A')}
- Recommended Position: {data.get('position_sizing', 'N/A')}

NEWS & SENTIMENT:
- Sentiment: {data.get('news_sentiment', 'neutral')}
- Market Impact: {data.get('market_impact', 'low')}
- Summary: {data.get('news_summary', 'No recent news')}

ANALYSIS REQUIREMENTS:
1. Provide a clear recommendation: BUY, SELL, or HOLD
2. Give a confidence level (0-1)
3. Explain your reasoning considering technical, fundamental, and sentiment factors
4. Assess the risk level and suitability for different investor types
5. Suggest an optimal position size if recommending BUY
6. Identify key risks and opportunities
7. Provide a price target or stop-loss level if applicable

Format your response as a structured analysis that considers all factors holistically.
"""
        return prompt
    
    def _parse_llm_response(self, response_text: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and structure the LLM response"""
        
        # Extract recommendation (BUY/SELL/HOLD)
        response_upper = response_text.upper()
        if "BUY" in response_upper and "SELL" not in response_upper:
            recommendation = "BUY"
        elif "SELL" in response_upper:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
        
        # Try to extract confidence level
        confidence = 0.7  # Default
        try:
            import re
            conf_match = re.search(r'confidence[:\s]*([0-9]*\.?[0-9]+)', response_text.lower())
            if conf_match:
                confidence = min(1.0, max(0.0, float(conf_match.group(1))))
        except:
            pass
        
        return {
            "ticker": data.get('current_price', 'N/A'),
            "recommendation": recommendation,
            "confidence": confidence,
            "reasoning": response_text,
            "risk_assessment": data.get('risk_category', 'Unknown'),
            "technical_data": {
                "rsi": data.get('rsi'),
                "beta": data.get('beta'),
                "volatility": data.get('volatility')
            },
            "position_sizing": data.get('position_sizing'),
            "sentiment": data.get('news_sentiment', 'neutral')
        }
    
    def compare_stocks(self, tickers: List[str], portfolio_value: float = 100000) -> Dict[str, Any]:
        """Compare multiple stocks and rank them"""
        print(f"🤖 Comparing stocks: {', '.join(tickers)}...")
        
        analyses = {}
        for ticker in tickers:
            analyses[ticker] = self.analyze_stock(ticker, portfolio_value)
        
        # Create comparison prompt
        comparison_prompt = f"""
Compare these stock analyses and rank them from best to worst investment opportunity:

{json.dumps(analyses, indent=2, default=str)}

Provide:
1. Ranking with brief justification
2. Best overall pick and why
3. Most conservative pick
4. Highest risk/reward pick
5. Portfolio allocation suggestions

Consider diversification, risk balance, and current market conditions.
"""
        
        try:
            response = self.model.generate_content(comparison_prompt)
            return {
                "comparison_analysis": response.text,
                "individual_analyses": analyses
            }
        except Exception as e:
            return {
                "comparison_analysis": f"Error in comparison: {e}",
                "individual_analyses": analyses
            }

# Example usage
if __name__ == "__main__":
    portfolio_manager = PortfolioManagerAgent()
    
    # Single stock analysis
    result = portfolio_manager.analyze_stock("AAPL", 100000)
    print(f"\n📊 PORTFOLIO MANAGER RECOMMENDATION:")
    print(f"Stock: AAPL")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Risk: {result['risk_assessment']}")
    print(f"\nReasoning:\n{result['reasoning']}")
    
    # Compare multiple stocks
    # comparison = portfolio_manager.compare_stocks(["AAPL", "GOOGL", "MSFT"], 100000)
    # print(f"\n📈 STOCK COMPARISON:")
    # print(comparison['comparison_analysis'])
