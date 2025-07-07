#!/usr/bin/env python3
"""
Market Scanner - Analyze Entire Stock Market and Provide Buy/Sell Recommendations

This module extends the Enhanced Finance Bot to scan hundreds of stocks,
rank them by potential, and provide actionable buy/sell recommendations.
"""

import os
import json
import time
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Any, Tuple
import pandas as pd
from portfolio_manager_rule_based import AdvancedPortfolioManagerAgent

class MarketScanner:
    """
    Comprehensive market scanner that analyzes multiple stocks and provides
    ranked buy/sell recommendations based on ML predictions and composite scoring.
    """
    
    def __init__(self, use_ml: bool = True, max_workers: int = 4):
        self.manager = AdvancedPortfolioManagerAgent(use_ml=use_ml)
        self.max_workers = max_workers
        self.results = []
        self.scan_timestamp = None
        
    def get_stock_universe(self, category: str = "sp500_sample") -> List[str]:
        """Get list of stocks to analyze"""
        stock_universes = {
            "sp500_sample": [
                # Large Cap Tech
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
                # Large Cap Traditional
                "JPM", "JNJ", "PG", "UNH", "HD", "V", "MA", "DIS",
                # Mid Cap Growth
                "PYPL", "ADBE", "CRM", "ZOOM", "SQ", "ROKU", "PTON", "SNAP",
                # Value Stocks
                "WMT", "XOM", "CVX", "KO", "PFE", "T", "VZ", "BAC",
                # Growth Stocks
                "AMD", "SHOP", "TWLO", "OKTA", "ZM", "DOCU", "CRWD", "NET"
            ],
            "large_cap": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM",
                "JNJ", "PG", "UNH", "HD", "V", "MA", "DIS", "WMT"
            ],
            "tech_focus": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
                "AMD", "ADBE", "CRM", "PYPL", "INTC", "ORCL", "CSCO", "IBM"
            ],
            "test_small": [
                "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"
            ]
        }
        
        return stock_universes.get(category, stock_universes["sp500_sample"])
    
    def analyze_single_stock(self, ticker: str) -> Dict[str, Any]:
        """Analyze a single stock and return results with error handling"""
        try:
            print(f"  Analyzing {ticker}...")
            result = self.manager.analyze_stock(ticker)
            
            # Add ticker and timestamp
            result["ticker"] = ticker
            result["analysis_time"] = datetime.now().isoformat()
            
            # Ensure numeric fields are present
            result["composite_score"] = result.get("composite_score", 0.5)
            result["confidence"] = result.get("confidence", 0.5)
            
            # Add ML prediction if available
            if result.get("ml_analysis") and "error" not in result.get("ml_analysis", {}):
                ml_data = result["ml_analysis"]
                result["ml_probability"] = ml_data.get("avg_prob_up", 0.5)
                result["ml_confidence"] = ml_data.get("confidence", 0.5)
            else:
                result["ml_probability"] = 0.5
                result["ml_confidence"] = 0.5
            
            # Check if we got a valid result
            if result.get("composite_score", 0) == 0 and result.get("confidence", 0) == 0:
                raise ValueError("No valid data returned")
            
            return result
            
        except Exception as e:
            print(f"    ❌ Error analyzing {ticker}: {str(e)[:100]}")
            return {
                "ticker": ticker,
                "recommendation": "HOLD",
                "confidence": 0.0,
                "composite_score": 0.5,
                "ml_probability": 0.5,
                "ml_confidence": 0.0,
                "error": str(e)[:200],  # Truncate long error messages
                "analysis_time": datetime.now().isoformat()
            }
    
    def scan_market(self, stock_universe: str = "sp500_sample", parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Scan the entire market and return ranked results
        """
        print(f"🔍 Starting Market Scan - {stock_universe.upper()}")
        print("=" * 60)
        
        tickers = self.get_stock_universe(stock_universe)
        self.scan_timestamp = datetime.now()
        
        print(f"📊 Analyzing {len(tickers)} stocks...")
        print(f"⚡ Parallel processing: {'Enabled' if parallel else 'Disabled'}")
        
        results = []
        
        if parallel and len(tickers) > 3:
            # Parallel processing for faster analysis
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_ticker = {executor.submit(self.analyze_single_stock, ticker): ticker for ticker in tickers}
                
                for future in concurrent.futures.as_completed(future_to_ticker):
                    result = future.result()
                    results.append(result)
        else:
            # Sequential processing
            for ticker in tickers:
                result = self.analyze_single_stock(ticker)
                results.append(result)
        
        # Filter out error results and sort
        valid_results = [r for r in results if "error" not in r]
        error_count = len(results) - len(valid_results)
        
        if error_count > 0:
            print(f"⚠️ {error_count} stocks had analysis errors")
        
        # Sort by composite score (descending)
        valid_results.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        
        self.results = valid_results
        
        print(f"✅ Market scan completed! Analyzed {len(valid_results)} stocks successfully")
        return valid_results
    
    def get_buy_recommendations(self, top_n: int = 10, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Get top buy recommendations"""
        if not self.results:
            raise ValueError("No scan results available. Run scan_market() first.")
        
        # Filter for buy recommendations with minimum confidence
        buy_candidates = [
            stock for stock in self.results
            if stock.get("recommendation") == "BUY" and 
               stock.get("confidence", 0) >= min_confidence
        ]
        
        # Sort by composite score and return top N
        buy_candidates.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        return buy_candidates[:top_n]
    
    def get_sell_recommendations(self, top_n: int = 10, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Get top sell recommendations"""
        if not self.results:
            raise ValueError("No scan results available. Run scan_market() first.")
        
        # Filter for sell recommendations with minimum confidence
        sell_candidates = [
            stock for stock in self.results
            if stock.get("recommendation") == "SELL" and 
               stock.get("confidence", 0) >= min_confidence
        ]
        
        # Sort by composite score (ascending for sells)
        sell_candidates.sort(key=lambda x: x.get("composite_score", 0))
        return sell_candidates[:top_n]
    
    def get_ml_top_picks(self, top_n: int = 10, min_ml_confidence: float = 0.7) -> List[Dict[str, Any]]:
        """Get top picks based on ML predictions"""
        if not self.results:
            raise ValueError("No scan results available. Run scan_market() first.")
        
        # Filter stocks with high ML prediction confidence
        ml_picks = [
            stock for stock in self.results
            if stock.get("ml_confidence", 0) >= min_ml_confidence and
               stock.get("ml_probability", 0.5) > 0.6  # Bullish ML prediction
        ]
        
        # Sort by ML probability of upward movement
        ml_picks.sort(key=lambda x: x.get("ml_probability", 0), reverse=True)
        return ml_picks[:top_n]
    
    def generate_market_report(self, save_to_file: bool = True) -> str:
        """Generate comprehensive market analysis report"""
        if not self.results:
            raise ValueError("No scan results available. Run scan_market() first.")
        
        # Get recommendations
        buy_recs = self.get_buy_recommendations()
        sell_recs = self.get_sell_recommendations()
        ml_picks = self.get_ml_top_picks()
        
        # Generate report
        report = f"""
🎯 MARKET ANALYSIS REPORT
{'='*50}
Generated: {self.scan_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Total Stocks Analyzed: {len(self.results)}

📈 TOP BUY RECOMMENDATIONS
{'-'*30}
"""
        
        for i, stock in enumerate(buy_recs[:5], 1):
            report += f"{i}. {stock['ticker']} - Score: {stock['composite_score']:.3f} - Confidence: {stock['confidence']:.1%}\n"
            report += f"   Reasoning: {stock.get('reasoning', 'N/A')[:100]}...\n\n"
        
        report += f"""
📉 TOP SELL RECOMMENDATIONS
{'-'*30}
"""
        
        for i, stock in enumerate(sell_recs[:5], 1):
            report += f"{i}. {stock['ticker']} - Score: {stock['composite_score']:.3f} - Confidence: {stock['confidence']:.1%}\n"
            report += f"   Reasoning: {stock.get('reasoning', 'N/A')[:100]}...\n\n"
        
        report += f"""
🤖 TOP ML PREDICTIONS (High Confidence)
{'-'*40}
"""
        
        for i, stock in enumerate(ml_picks[:5], 1):
            ml_accuracy = stock.get('ml_analysis', {}).get('average_accuracy', 0) if stock.get('ml_analysis') else 0
            accuracy_text = f" (Acc: {ml_accuracy:.1%})" if ml_accuracy > 0 else ""
            report += f"{i}. {stock['ticker']} - ML Prob: {stock['ml_probability']:.1%} - ML Conf: {stock['ml_confidence']:.1%}{accuracy_text}\n"
        
        # Market summary statistics
        avg_score = sum(s.get('composite_score', 0) for s in self.results) / len(self.results)
        buy_count = len([s for s in self.results if s.get('recommendation') == 'BUY'])
        sell_count = len([s for s in self.results if s.get('recommendation') == 'SELL'])
        hold_count = len(self.results) - buy_count - sell_count
        
        report += f"""

📊 MARKET SUMMARY
{'-'*20}
Average Composite Score: {avg_score:.3f}
Buy Signals: {buy_count} ({buy_count/len(self.results)*100:.1f}%)
Sell Signals: {sell_count} ({sell_count/len(self.results)*100:.1f}%)
Hold Signals: {hold_count} ({hold_count/len(self.results)*100:.1f}%)

⚠️ DISCLAIMER
This analysis is for educational purposes only. Past performance does not guarantee future results.
Always conduct your own research and consider your risk tolerance before making investment decisions.
"""
        
        if save_to_file:
            filename = f"market_report_{self.scan_timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w') as f:
                f.write(report)
            print(f"📄 Report saved to {filename}")
        
        return report
    
    def save_results_to_csv(self, filename: str = None) -> str:
        """Save detailed results to CSV for further analysis"""
        if not self.results:
            raise ValueError("No scan results available. Run scan_market() first.")
        
        if filename is None:
            filename = f"market_scan_{self.scan_timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Prepare data for CSV
        csv_data = []
        for stock in self.results:
            csv_data.append({
                'ticker': stock.get('ticker'),
                'recommendation': stock.get('recommendation'),
                'confidence': stock.get('confidence'),
                'composite_score': stock.get('composite_score'),
                'ml_probability': stock.get('ml_probability'),
                'ml_confidence': stock.get('ml_confidence'),
                'ml_avg_accuracy': stock.get('ml_analysis', {}).get('average_accuracy', 0) if stock.get('ml_analysis') else 0,
                'risk_assessment': stock.get('risk_assessment'),
                'reasoning': stock.get('reasoning', '')[:200]  # Truncate for CSV
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False)
        
        print(f"💾 Results saved to {filename}")
        return filename


def main():
    """Demo of the market scanner"""
    print("🚀 Enhanced Finance Bot - Market Scanner")
    print("=" * 50)
    
    # Initialize scanner
    scanner = MarketScanner(use_ml=True, max_workers=3)
    
    # Choose stock universe
    print("\nSelect stock universe:")
    print("1. Test Small (5 stocks) - Quick test")
    print("2. Large Cap (16 stocks) - Major companies")
    print("3. Tech Focus (16 stocks) - Technology companies")
    print("4. SP500 Sample (40 stocks) - Diverse sample")
    
    choice = input("Enter choice (1-4, default=1): ").strip() or "1"
    
    universe_map = {
        "1": "test_small",
        "2": "large_cap", 
        "3": "tech_focus",
        "4": "sp500_sample"
    }
    
    universe = universe_map.get(choice, "test_small")
    
    # Run market scan
    try:
        results = scanner.scan_market(universe, parallel=True)
        
        # Generate and display report
        report = scanner.generate_market_report(save_to_file=True)
        print(report)
        
        # Save detailed results
        scanner.save_results_to_csv()
        
        # Interactive recommendations
        print("\n" + "=" * 50)
        print("🎯 QUICK RECOMMENDATIONS")
        print("=" * 50)
        
        buy_recs = scanner.get_buy_recommendations(top_n=3)
        if buy_recs:
            print("\n🟢 TOP 3 BUY CANDIDATES:")
            for i, stock in enumerate(buy_recs, 1):
                print(f"  {i}. {stock['ticker']} - Score: {stock['composite_score']:.3f} - Confidence: {stock['confidence']:.1%}")
        
        sell_recs = scanner.get_sell_recommendations(top_n=3)
        if sell_recs:
            print("\n🔴 TOP 3 SELL CANDIDATES:")
            for i, stock in enumerate(sell_recs, 1):
                print(f"  {i}. {stock['ticker']} - Score: {stock['composite_score']:.3f} - Confidence: {stock['confidence']:.1%}")
        
        ml_picks = scanner.get_ml_top_picks(top_n=3, min_ml_confidence=0.6)
        if ml_picks:
            print("\n🤖 TOP 3 ML PREDICTIONS:")
            for i, stock in enumerate(ml_picks, 1):
                print(f"  {i}. {stock['ticker']} - ML Prob: {stock['ml_probability']:.1%} - Confidence: {stock['ml_confidence']:.1%}")
        
        print(f"\n✅ Market scan completed! Check the generated files for detailed results.")
        
    except Exception as e:
        print(f"❌ Error during market scan: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
