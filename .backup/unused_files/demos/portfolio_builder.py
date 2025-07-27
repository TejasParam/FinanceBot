#!/usr/bin/env python3
"""
Portfolio Builder - Construct Optimized Portfolios from Market Analysis

This module builds on the market scanner to create diversified, optimized portfolios
based on ML predictions, risk metrics, and modern portfolio theory principles.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime
import json

# Handle imports for both direct execution and module import
try:
    from market_scanner import MarketScanner
except ImportError:
    from .market_scanner import MarketScanner

class PortfolioBuilder:
    """
    Advanced portfolio construction using market analysis results
    """
    
    def __init__(self, market_scanner: MarketScanner = None):
        self.scanner = market_scanner or MarketScanner(use_ml=True)
        self.portfolio_results = {}
        
    def build_growth_portfolio(self, portfolio_value: float = 100000, max_positions: int = 10) -> Dict[str, Any]:
        """Build a growth-focused portfolio"""
        if not self.scanner.results:
            raise ValueError("No market scan results available. Run scanner.scan_market() first.")
        
        print(f"🚀 Building Growth Portfolio (${portfolio_value:,.0f})")
        
        # Get high-growth candidates (more lenient criteria)
        growth_candidates = [
            stock for stock in self.scanner.results
            if stock.get('composite_score', 0) > 0.5 and  # Lowered from 0.6
               stock.get('confidence', 0) > 0.5 and       # Lowered from high threshold
               stock.get('ml_probability', 0.5) > 0.55    # Lowered from 0.65
        ]
        
        # If still no candidates, use top scoring stocks regardless of recommendation
        if not growth_candidates:
            print("  📊 Using top-scoring stocks for growth portfolio...")
            all_stocks = sorted(self.scanner.results, key=lambda x: x.get('composite_score', 0), reverse=True)
            growth_candidates = all_stocks[:max_positions]
        
        # Sort by composite score and ML probability
        growth_candidates.sort(
            key=lambda x: (x.get('composite_score', 0) * 0.6 + x.get('ml_probability', 0.5) * 0.4), 
            reverse=True
        )
        
        # Select top positions
        selected_stocks = growth_candidates[:max_positions]
        
        if not selected_stocks:
            return {"error": "No suitable growth stocks found"}
        
        # Equal weight allocation (can be enhanced with optimization)
        weight_per_stock = 1.0 / len(selected_stocks)
        allocation_per_stock = portfolio_value * weight_per_stock
        
        portfolio = {
            "strategy": "Growth",
            "total_value": portfolio_value,
            "positions": [],
            "metrics": {}
        }
        
        total_score = 0
        total_ml_prob = 0
        
        for stock in selected_stocks:
            position = {
                "ticker": stock['ticker'],
                "allocation": allocation_per_stock,
                "weight": weight_per_stock,
                "recommendation": stock['recommendation'],
                "composite_score": stock.get('composite_score', 0),
                "ml_probability": stock.get('ml_probability', 0.5),
                "confidence": stock.get('confidence', 0),
                "reasoning": stock.get('reasoning', '')[:100]
            }
            portfolio["positions"].append(position)
            total_score += stock.get('composite_score', 0)
            total_ml_prob += stock.get('ml_probability', 0.5)
        
        # Portfolio metrics
        portfolio["metrics"] = {
            "avg_composite_score": total_score / len(selected_stocks),
            "avg_ml_probability": total_ml_prob / len(selected_stocks),
            "num_positions": len(selected_stocks),
            "cash_allocation": 0,
            "risk_level": "High"
        }
        
        return portfolio
    
    def build_balanced_portfolio(self, portfolio_value: float = 100000, max_positions: int = 15) -> Dict[str, Any]:
        """Build a balanced portfolio with growth and value stocks"""
        if not self.scanner.results:
            raise ValueError("No market scan results available. Run scanner.scan_market() first.")
        
        print(f"⚖️ Building Balanced Portfolio (${portfolio_value:,.0f})")
        
        # Get buy candidates (adjusted threshold)
        buy_candidates = [
            stock for stock in self.scanner.results
            if stock.get('recommendation') == 'BUY' and 
               stock.get('confidence', 0) > 0.5
        ]
        
        # Separate growth and value (adjusted thresholds)
        growth_stocks = [s for s in buy_candidates if s.get('composite_score', 0) > 0.57]
        value_stocks = [s for s in buy_candidates if s.get('composite_score', 0) <= 0.57 and s.get('composite_score', 0) > 0.50]
        
        # Allocate 60% to growth, 40% to value
        growth_allocation = portfolio_value * 0.6
        value_allocation = portfolio_value * 0.4
        
        # Select stocks
        growth_count = min(len(growth_stocks), max_positions // 2)
        value_count = min(len(value_stocks), max_positions - growth_count)
        
        selected_growth = sorted(growth_stocks, key=lambda x: x.get('composite_score', 0), reverse=True)[:growth_count]
        selected_value = sorted(value_stocks, key=lambda x: x.get('composite_score', 0), reverse=True)[:value_count]
        
        portfolio = {
            "strategy": "Balanced",
            "total_value": portfolio_value,
            "positions": [],
            "metrics": {}
        }
        
        # Add growth positions
        if selected_growth:
            weight_per_growth = 0.6 / len(selected_growth)
            allocation_per_growth = growth_allocation / len(selected_growth)
            
            for stock in selected_growth:
                position = {
                    "ticker": stock['ticker'],
                    "allocation": allocation_per_growth,
                    "weight": weight_per_growth,
                    "category": "Growth",
                    "composite_score": stock.get('composite_score', 0),
                    "ml_probability": stock.get('ml_probability', 0.5)
                }
                portfolio["positions"].append(position)
        
        # Add value positions
        if selected_value:
            weight_per_value = 0.4 / len(selected_value)
            allocation_per_value = value_allocation / len(selected_value)
            
            for stock in selected_value:
                position = {
                    "ticker": stock['ticker'],
                    "allocation": allocation_per_value,
                    "weight": weight_per_value,
                    "category": "Value",
                    "composite_score": stock.get('composite_score', 0),
                    "ml_probability": stock.get('ml_probability', 0.5)
                }
                portfolio["positions"].append(position)
        
        # Calculate metrics
        if portfolio["positions"]:
            avg_score = sum(p['composite_score'] for p in portfolio["positions"]) / len(portfolio["positions"])
            avg_ml_prob = sum(p['ml_probability'] for p in portfolio["positions"]) / len(portfolio["positions"])
            
            portfolio["metrics"] = {
                "avg_composite_score": avg_score,
                "avg_ml_probability": avg_ml_prob,
                "num_positions": len(portfolio["positions"]),
                "growth_positions": len(selected_growth),
                "value_positions": len(selected_value),
                "cash_allocation": 0,
                "risk_level": "Medium"
            }
        
        return portfolio
    
    def build_conservative_portfolio(self, portfolio_value: float = 100000, cash_percentage: float = 0.2) -> Dict[str, Any]:
        """Build a conservative portfolio with cash buffer"""
        if not self.scanner.results:
            raise ValueError("No market scan results available. Run scanner.scan_market() first.")
        
        print(f"🛡️ Building Conservative Portfolio (${portfolio_value:,.0f})")
        
        # Conservative selection criteria (adjusted for more realistic thresholds)
        conservative_candidates = [
            stock for stock in self.scanner.results
            if stock.get('recommendation') in ['BUY', 'HOLD'] and 
               stock.get('confidence', 0) > 0.5 and
               stock.get('composite_score', 0) > 0.50 and
               stock.get('risk_assessment', 'High') in ['Low', 'Medium', 'Moderate Risk']
        ]
        
        # Sort by confidence and composite score
        conservative_candidates.sort(
            key=lambda x: (x.get('confidence', 0) * 0.5 + x.get('composite_score', 0) * 0.5), 
            reverse=True
        )
        
        # Allocate funds
        cash_allocation = portfolio_value * cash_percentage
        stock_allocation = portfolio_value * (1 - cash_percentage)
        
        # Select top conservative picks
        max_positions = 8  # More concentrated for conservative approach
        selected_stocks = conservative_candidates[:max_positions]
        
        portfolio = {
            "strategy": "Conservative",
            "total_value": portfolio_value,
            "positions": [],
            "metrics": {}
        }
        
        if selected_stocks:
            weight_per_stock = (1 - cash_percentage) / len(selected_stocks)
            allocation_per_stock = stock_allocation / len(selected_stocks)
            
            for stock in selected_stocks:
                position = {
                    "ticker": stock['ticker'],
                    "allocation": allocation_per_stock,
                    "weight": weight_per_stock,
                    "recommendation": stock['recommendation'],
                    "composite_score": stock.get('composite_score', 0),
                    "confidence": stock.get('confidence', 0),
                    "risk_assessment": stock.get('risk_assessment', 'Unknown')
                }
                portfolio["positions"].append(position)
        
        # Add cash position
        portfolio["positions"].append({
            "ticker": "CASH",
            "allocation": cash_allocation,
            "weight": cash_percentage,
            "category": "Cash"
        })
        
        # Calculate metrics
        if len(portfolio["positions"]) > 1:  # Exclude cash for stock metrics
            stock_positions = [p for p in portfolio["positions"] if p["ticker"] != "CASH"]
            avg_score = sum(p['composite_score'] for p in stock_positions) / len(stock_positions)
            avg_confidence = sum(p['confidence'] for p in stock_positions) / len(stock_positions)
            
            portfolio["metrics"] = {
                "avg_composite_score": avg_score,
                "avg_confidence": avg_confidence,
                "num_stock_positions": len(stock_positions),
                "cash_allocation": cash_percentage,
                "risk_level": "Low"
            }
        
        return portfolio
    
    def generate_portfolio_report(self, portfolios: List[Dict[str, Any]], save_to_file: bool = True) -> str:
        """Generate comprehensive portfolio comparison report"""
        timestamp = datetime.now()
        
        report = f"""
📊 PORTFOLIO ANALYSIS REPORT
{'='*60}
Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        for portfolio in portfolios:
            strategy = portfolio.get('strategy', 'Unknown')
            total_value = portfolio.get('total_value', 0)
            positions = portfolio.get('positions', [])
            metrics = portfolio.get('metrics', {})
            
            report += f"""
🎯 {strategy.upper()} PORTFOLIO (${total_value:,.0f})
{'-'*40}
"""
            
            if 'error' in portfolio:
                report += f"❌ Error: {portfolio['error']}\n\n"
                continue
            
            # Portfolio composition
            stock_positions = [p for p in positions if p.get('ticker') != 'CASH']
            cash_position = next((p for p in positions if p.get('ticker') == 'CASH'), None)
            
            report += f"Number of Positions: {len(stock_positions)}\n"
            if cash_position:
                report += f"Cash Allocation: ${cash_position['allocation']:,.0f} ({cash_position['weight']:.1%})\n"
            
            # Key metrics
            if metrics:
                report += f"Average Composite Score: {metrics.get('avg_composite_score', 0):.3f}\n"
                if 'avg_ml_probability' in metrics:
                    report += f"Average ML Probability: {metrics.get('avg_ml_probability', 0):.1%}\n"
                if 'avg_confidence' in metrics:
                    report += f"Average Confidence: {metrics.get('avg_confidence', 0):.1%}\n"
                report += f"Risk Level: {metrics.get('risk_level', 'Unknown')}\n"
            
            # Top positions
            report += f"\nTop Holdings:\n"
            for i, position in enumerate(stock_positions[:5], 1):
                ticker = position['ticker']
                allocation = position['allocation']
                weight = position['weight']
                score = position.get('composite_score', 0)
                report += f"  {i}. {ticker}: ${allocation:,.0f} ({weight:.1%}) - Score: {score:.3f}\n"
            
            if len(stock_positions) > 5:
                report += f"  ... and {len(stock_positions) - 5} more positions\n"
            
            report += "\n"
        
        # Portfolio comparison
        if len(portfolios) > 1:
            report += f"""
📈 PORTFOLIO COMPARISON
{'-'*30}
"""
            for portfolio in portfolios:
                if 'error' not in portfolio:
                    strategy = portfolio.get('strategy', 'Unknown')
                    metrics = portfolio.get('metrics', {})
                    avg_score = metrics.get('avg_composite_score', 0)
                    risk_level = metrics.get('risk_level', 'Unknown')
                    num_positions = metrics.get('num_positions', metrics.get('num_stock_positions', 0))
                    
                    report += f"{strategy}: Score {avg_score:.3f}, Risk {risk_level}, {num_positions} positions\n"
        
        report += f"""

⚠️ IMPORTANT DISCLAIMERS
{'-'*30}
• These are model-generated recommendations for educational purposes only
• Past performance does not guarantee future results
• All investments carry risk of loss
• Diversification does not guarantee profits or protect against losses
• Consult with a financial advisor before making investment decisions
• Consider your risk tolerance, investment objectives, and time horizon
"""
        
        if save_to_file:
            filename = f"portfolio_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w') as f:
                f.write(report)
            print(f"📄 Portfolio report saved to {filename}")
        
        return report
    
    def save_portfolios_to_json(self, portfolios: List[Dict[str, Any]], filename: str = None) -> str:
        """Save portfolio data to JSON for further analysis"""
        if filename is None:
            timestamp = datetime.now()
            filename = f"portfolios_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare data
        portfolio_data = {
            "generated_at": datetime.now().isoformat(),
            "portfolios": portfolios
        }
        
        with open(filename, 'w') as f:
            json.dump(portfolio_data, f, indent=2)
        
        print(f"💾 Portfolio data saved to {filename}")
        return filename


def main():
    """Demo of the portfolio builder"""
    print("🏦 Enhanced Finance Bot - Portfolio Builder")
    print("=" * 60)
    
    # Initialize components
    scanner = MarketScanner(use_ml=True, max_workers=3)
    builder = PortfolioBuilder(scanner)
    
    # Get portfolio value
    portfolio_value_input = input("Enter portfolio value (default: $100,000): ").strip()
    try:
        portfolio_value = float(portfolio_value_input.replace('$', '').replace(',', '')) if portfolio_value_input else 100000
    except ValueError:
        portfolio_value = 100000
    
    print(f"\n💰 Building portfolios with ${portfolio_value:,.0f}")
    
    # Choose stock universe for analysis
    print("\nSelect stock universe for analysis:")
    print("1. Test Small (5 stocks) - Quick demo")
    print("2. Large Cap (16 stocks) - Major companies")
    print("3. SP500 Sample (40 stocks) - Comprehensive analysis")
    
    choice = input("Enter choice (1-3, default=1): ").strip() or "1"
    
    universe_map = {"1": "test_small", "2": "large_cap", "3": "sp500_sample"}
    universe = universe_map.get(choice, "test_small")
    
    try:
        # Run market scan first
        print(f"\n🔍 Scanning market ({universe})...")
        scanner.scan_market(universe, parallel=True)
        
        # Build different portfolio strategies
        print(f"\n🏗️ Building portfolio strategies...")
        
        portfolios = []
        
        # Growth portfolio
        growth_portfolio = builder.build_growth_portfolio(portfolio_value)
        if 'error' not in growth_portfolio:
            portfolios.append(growth_portfolio)
        
        # Balanced portfolio
        balanced_portfolio = builder.build_balanced_portfolio(portfolio_value)
        if 'error' not in balanced_portfolio:
            portfolios.append(balanced_portfolio)
        
        # Conservative portfolio
        conservative_portfolio = builder.build_conservative_portfolio(portfolio_value)
        if 'error' not in conservative_portfolio:
            portfolios.append(conservative_portfolio)
        
        if not portfolios:
            print("❌ No suitable portfolios could be built with current market conditions.")
            return
        
        # Generate and display report
        report = builder.generate_portfolio_report(portfolios, save_to_file=True)
        print(report)
        
        # Save portfolio data
        builder.save_portfolios_to_json(portfolios)
        
        print(f"\n✅ Portfolio analysis completed!")
        print(f"📊 Generated {len(portfolios)} portfolio strategies")
        print(f"📁 Check the generated files for detailed results")
        
    except Exception as e:
        print(f"❌ Error during portfolio building: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
