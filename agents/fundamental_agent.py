"""
Fundamental Analysis Agent for financial metrics and valuation analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
import random
import time
from .base_agent import BaseAgent

class FundamentalAnalysisAgent(BaseAgent):
    """
    Agent specialized in fundamental analysis of stocks.
    Analyzes financial metrics, valuation ratios, and company fundamentals.
    """
    
    def __init__(self):
        super().__init__("FundamentalAnalysis")
        
    def analyze(self, ticker: str, **kwargs) -> Dict[str, Any]:
        """
        Perform fundamental analysis for the given ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with fundamental analysis results
        """
        try:
            # Get fundamental data
            financial_metrics = self._get_financial_metrics(ticker)
            valuation_metrics = self._get_valuation_metrics(ticker)
            growth_metrics = self._get_growth_metrics(ticker)
            quality_metrics = self._get_quality_metrics(ticker)
            
            # Calculate overall fundamental score
            fundamental_score = self._calculate_fundamental_score(
                financial_metrics, valuation_metrics, growth_metrics, quality_metrics
            )
            
            # Calculate confidence based on data availability
            confidence = self._calculate_fundamental_confidence(
                financial_metrics, valuation_metrics, growth_metrics, quality_metrics
            )
            
            # Generate reasoning
            reasoning = self._generate_fundamental_reasoning(
                fundamental_score, financial_metrics, valuation_metrics, 
                growth_metrics, quality_metrics
            )
            
            return {
                'score': fundamental_score,
                'confidence': confidence,
                'reasoning': reasoning,
                'breakdown': {
                    'financial': financial_metrics,
                    'valuation': valuation_metrics,
                    'growth': growth_metrics,
                    'quality': quality_metrics
                },
                'overall_rating': self._get_rating_from_score(fundamental_score)
            }
            
        except Exception as e:
            return {
                'error': f'Fundamental analysis failed: {str(e)}',
                'score': 0.0,
                'confidence': 0.0,
                'reasoning': f'Fundamental analysis error: {str(e)[:100]}'
            }
    
    def _get_financial_metrics(self, ticker: str) -> Dict[str, Any]:
        """Get financial strength metrics (simulated for demo)"""
        try:
            # In a real implementation, this would fetch actual financial data
            # from sources like Yahoo Finance, Alpha Vantage, or financial APIs
            
            # Simulate financial metrics with some ticker-specific biases
            base_metrics = {
                'revenue_growth': random.uniform(-0.1, 0.3),
                'profit_margin': random.uniform(0.05, 0.25),
                'debt_to_equity': random.uniform(0.1, 1.5),
                'current_ratio': random.uniform(1.0, 3.0),
                'roa': random.uniform(0.02, 0.15),  # Return on Assets
                'roe': random.uniform(0.05, 0.25),  # Return on Equity
            }
            
            # Add some company-specific adjustments for realism
            if ticker in ['AAPL', 'MSFT', 'GOOGL']:
                base_metrics['profit_margin'] *= 1.3  # Tech giants have higher margins
                base_metrics['roa'] *= 1.2
                base_metrics['debt_to_equity'] *= 0.7  # Lower debt
            elif ticker in ['TSLA']:
                base_metrics['revenue_growth'] *= 1.5  # Higher growth
                base_metrics['debt_to_equity'] *= 1.3  # Higher debt for growth
            
            return base_metrics
            
        except Exception as e:
            return {'error': f'Failed to get financial metrics: {str(e)}'}
    
    def _get_valuation_metrics(self, ticker: str) -> Dict[str, Any]:
        """Get valuation metrics (simulated for demo)"""
        try:
            # Simulate valuation metrics
            base_metrics = {
                'pe_ratio': random.uniform(10, 40),
                'peg_ratio': random.uniform(0.5, 3.0),
                'price_to_book': random.uniform(1.0, 5.0),
                'price_to_sales': random.uniform(1.0, 8.0),
                'ev_ebitda': random.uniform(8, 25),
                'dividend_yield': random.uniform(0.0, 0.06)
            }
            
            # Sector-specific adjustments
            if ticker in ['AAPL', 'MSFT', 'GOOGL']:
                base_metrics['pe_ratio'] *= 1.2  # Tech premium
                base_metrics['price_to_sales'] *= 1.3
            elif ticker in ['TSLA', 'NVDA']:
                base_metrics['pe_ratio'] *= 2.0  # Growth premium
                base_metrics['peg_ratio'] *= 1.5
                base_metrics['dividend_yield'] = 0.0  # Growth stocks often don't pay dividends
            
            return base_metrics
            
        except Exception as e:
            return {'error': f'Failed to get valuation metrics: {str(e)}'}
    
    def _get_growth_metrics(self, ticker: str) -> Dict[str, Any]:
        """Get growth metrics (simulated for demo)"""
        try:
            base_metrics = {
                'revenue_growth_3y': random.uniform(-0.05, 0.25),
                'earnings_growth_3y': random.uniform(-0.1, 0.35),
                'book_value_growth': random.uniform(0.0, 0.20),
                'dividend_growth': random.uniform(0.0, 0.15),
                'projected_growth': random.uniform(-0.05, 0.30)
            }
            
            # Growth company adjustments
            if ticker in ['TSLA', 'NVDA']:
                for key in base_metrics:
                    if 'growth' in key:
                        base_metrics[key] *= 1.8
            elif ticker in ['AAPL', 'MSFT']:
                for key in base_metrics:
                    if 'growth' in key:
                        base_metrics[key] *= 1.2
            
            return base_metrics
            
        except Exception as e:
            return {'error': f'Failed to get growth metrics: {str(e)}'}
    
    def _get_quality_metrics(self, ticker: str) -> Dict[str, Any]:
        """Get quality/stability metrics (simulated for demo)"""
        try:
            base_metrics = {
                'earnings_stability': random.uniform(0.3, 0.9),
                'revenue_stability': random.uniform(0.4, 0.9),
                'dividend_consistency': random.uniform(0.0, 1.0),
                'balance_sheet_strength': random.uniform(0.3, 0.9),
                'management_effectiveness': random.uniform(0.4, 0.9)
            }
            
            # Blue chip adjustments
            if ticker in ['AAPL', 'MSFT', 'JNJ', 'PG']:
                for key in base_metrics:
                    base_metrics[key] = min(1.0, base_metrics[key] * 1.2)
            
            return base_metrics
            
        except Exception as e:
            return {'error': f'Failed to get quality metrics: {str(e)}'}
    
    def _calculate_fundamental_score(self, financial: Dict, valuation: Dict, 
                                   growth: Dict, quality: Dict) -> float:
        """Calculate overall fundamental score from -1 to 1"""
        scores = []
        weights = []
        
        # Financial strength score (25% weight)
        if 'error' not in financial:
            fin_score = 0.0
            if financial.get('profit_margin', 0) > 0.15:
                fin_score += 0.3
            if financial.get('roa', 0) > 0.08:
                fin_score += 0.3
            if financial.get('debt_to_equity', 1.0) < 0.5:
                fin_score += 0.2
            if financial.get('current_ratio', 1.0) > 1.5:
                fin_score += 0.2
            
            scores.append(fin_score - 0.5)  # Center around 0
            weights.append(0.25)
        
        # Valuation score (30% weight)
        if 'error' not in valuation:
            val_score = 0.0
            pe_ratio = valuation.get('pe_ratio', 20)
            if pe_ratio < 15:
                val_score += 0.4  # Cheap
            elif pe_ratio > 30:
                val_score -= 0.3  # Expensive
            
            peg_ratio = valuation.get('peg_ratio', 1.5)
            if peg_ratio < 1.0:
                val_score += 0.3
            elif peg_ratio > 2.0:
                val_score -= 0.3
            
            price_to_book = valuation.get('price_to_book', 2.0)
            if price_to_book < 1.5:
                val_score += 0.2
            elif price_to_book > 3.0:
                val_score -= 0.2
            
            dividend_yield = valuation.get('dividend_yield', 0)
            if dividend_yield > 0.03:
                val_score += 0.1
            
            scores.append(max(-1, min(1, val_score)))
            weights.append(0.30)
        
        # Growth score (25% weight)
        if 'error' not in growth:
            growth_score = 0.0
            revenue_growth = growth.get('revenue_growth_3y', 0)
            earnings_growth = growth.get('earnings_growth_3y', 0)
            
            if revenue_growth > 0.15:
                growth_score += 0.4
            elif revenue_growth < 0:
                growth_score -= 0.3
            
            if earnings_growth > 0.20:
                growth_score += 0.4
            elif earnings_growth < 0:
                growth_score -= 0.3
            
            projected_growth = growth.get('projected_growth', 0)
            if projected_growth > 0.15:
                growth_score += 0.2
            
            scores.append(max(-1, min(1, growth_score)))
            weights.append(0.25)
        
        # Quality score (20% weight)
        if 'error' not in quality:
            qual_score = 0.0
            for metric in ['earnings_stability', 'revenue_stability', 
                          'balance_sheet_strength', 'management_effectiveness']:
                value = quality.get(metric, 0.5)
                qual_score += (value - 0.5) * 0.25
            
            scores.append(max(-1, min(1, qual_score)))
            weights.append(0.20)
        
        if not scores:
            return 0.0
        
        # Weighted average
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        return max(-1.0, min(1.0, weighted_score))
    
    def _calculate_fundamental_confidence(self, financial: Dict, valuation: Dict,
                                        growth: Dict, quality: Dict) -> float:
        """Calculate confidence in fundamental analysis"""
        available_data = sum(1 for data in [financial, valuation, growth, quality] 
                           if 'error' not in data)
        
        base_confidence = available_data / 4.0
        
        # Boost confidence if we have comprehensive data
        if available_data >= 3:
            base_confidence *= 1.2
        
        return min(1.0, base_confidence)
    
    def _generate_fundamental_reasoning(self, score: float, financial: Dict, 
                                      valuation: Dict, growth: Dict, quality: Dict) -> str:
        """Generate human-readable reasoning for fundamental analysis"""
        reasoning_parts = []
        
        # Overall assessment
        if score > 0.3:
            reasoning_parts.append("Strong fundamental outlook.")
        elif score < -0.3:
            reasoning_parts.append("Weak fundamental outlook.")
        else:
            reasoning_parts.append("Mixed fundamental signals.")
        
        # Financial strength
        if 'error' not in financial:
            profit_margin = financial.get('profit_margin', 0)
            debt_to_equity = financial.get('debt_to_equity', 1.0)
            
            if profit_margin > 0.15:
                reasoning_parts.append(f"Strong profitability ({profit_margin:.1%} margin).")
            elif profit_margin < 0.05:
                reasoning_parts.append(f"Low profitability ({profit_margin:.1%} margin).")
            
            if debt_to_equity < 0.3:
                reasoning_parts.append("Low debt levels.")
            elif debt_to_equity > 1.0:
                reasoning_parts.append("High debt levels.")
        
        # Valuation
        if 'error' not in valuation:
            pe_ratio = valuation.get('pe_ratio', 20)
            peg_ratio = valuation.get('peg_ratio', 1.5)
            
            if pe_ratio < 15:
                reasoning_parts.append(f"Attractive valuation (P/E: {pe_ratio:.1f}).")
            elif pe_ratio > 30:
                reasoning_parts.append(f"High valuation (P/E: {pe_ratio:.1f}).")
            
            if peg_ratio < 1.0:
                reasoning_parts.append(f"Growth at reasonable price (PEG: {peg_ratio:.1f}).")
            elif peg_ratio > 2.0:
                reasoning_parts.append(f"Expensive relative to growth (PEG: {peg_ratio:.1f}).")
        
        # Growth
        if 'error' not in growth:
            revenue_growth = growth.get('revenue_growth_3y', 0)
            earnings_growth = growth.get('earnings_growth_3y', 0)
            
            if revenue_growth > 0.15:
                reasoning_parts.append(f"Strong revenue growth ({revenue_growth:.1%}).")
            elif revenue_growth < 0:
                reasoning_parts.append(f"Declining revenue ({revenue_growth:.1%}).")
            
            if earnings_growth > 0.20:
                reasoning_parts.append(f"Excellent earnings growth ({earnings_growth:.1%}).")
        
        return " ".join(reasoning_parts)
    
    def _get_rating_from_score(self, score: float) -> str:
        """Convert score to rating"""
        if score > 0.6:
            return "Strong Buy"
        elif score > 0.2:
            return "Buy"
        elif score > -0.2:
            return "Hold"
        elif score > -0.6:
            return "Sell"
        else:
            return "Strong Sell"
