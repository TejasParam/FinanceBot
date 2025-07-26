"""
Sentiment Analysis Agent for financial news and social media sentiment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
from .base_agent import BaseAgent
from news_sentiment_enhanced import EnhancedNewsSentimentAnalyzer
import requests
import time
import random

class SentimentAnalysisAgent(BaseAgent):
    """
    Agent specialized in sentiment analysis from news and social media.
    Analyzes public sentiment to gauge market sentiment for stocks.
    """
    
    def __init__(self):
        super().__init__("SentimentAnalysis")
        try:
            self.news_analyzer = EnhancedNewsSentimentAnalyzer()
        except Exception as e:
            print(f"Warning: Could not initialize news analyzer: {e}")
            self.news_analyzer = None
        
    def analyze(self, ticker: str, **kwargs) -> Dict[str, Any]:
        """
        Perform sentiment analysis for the given ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Use existing news analyst if available
            if self.news_analyzer:
                try:
                    news_data = self.news_analyzer.analyze_stock_sentiment(ticker)
                    return self._process_news_data(ticker, news_data)
                except Exception as e:
                    print(f"News analyzer failed, using fallback: {e}")
            
            # Fallback to simulated sentiment analysis
            return self._simulate_sentiment_analysis(ticker)
            
        except Exception as e:
            return {
                'error': f'Sentiment analysis failed: {str(e)}',
                'score': 0.0,
                'confidence': 0.0,
                'reasoning': f'Sentiment analysis error: {str(e)[:100]}'
            }
    
    def _process_news_data(self, ticker: str, news_data) -> Dict[str, Any]:
        """Process news data from the news analyzer"""
        try:
            if isinstance(news_data, list) and len(news_data) > 0:
                # Calculate overall sentiment from news items
                sentiments = []
                for item in news_data:
                    sentiment = item.get('sentiment', 'neutral')
                    if sentiment == 'positive':
                        sentiments.append(0.7)
                    elif sentiment == 'very_positive':
                        sentiments.append(0.9)
                    elif sentiment == 'negative':
                        sentiments.append(0.3)
                    elif sentiment == 'very_negative':
                        sentiments.append(0.1)
                    else:
                        sentiments.append(0.5)
                
                avg_sentiment = sum(sentiments) / len(sentiments)
                confidence = min(0.9, 0.5 + (len(news_data) / 20))
                
                return {
                    'score': avg_sentiment,
                    'confidence': confidence,
                    'reasoning': f'Analyzed {len(news_data)} news items with average sentiment',
                    'articles_analyzed': len(news_data),
                    'sentiment_distribution': {
                        'positive': sum(1 for s in sentiments if s > 0.6),
                        'neutral': sum(1 for s in sentiments if 0.4 <= s <= 0.6),
                        'negative': sum(1 for s in sentiments if s < 0.4)
                    }
                }
            else:
                return self._simulate_sentiment_analysis(ticker)
                
        except Exception as e:
            return self._simulate_sentiment_analysis(ticker)
    
    def _analyze_news_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze sentiment from financial news (simulated for demo)"""
        try:
            # In a real implementation, this would:
            # 1. Fetch recent news articles about the ticker
            # 2. Use NLP/sentiment models to analyze text
            # 3. Return aggregated sentiment score
            
            # For demo purposes, simulate news sentiment with some randomness
            # but bias towards slightly positive (markets tend to be optimistic)
            base_sentiment = random.uniform(-0.3, 0.5)
            
            # Add some company-specific biases for demo
            if ticker in ['AAPL', 'MSFT', 'GOOGL']:
                base_sentiment += 0.1  # Large caps tend to have more positive coverage
            elif ticker in ['TSLA', 'NVDA']:
                base_sentiment += random.uniform(-0.2, 0.3)  # More volatile sentiment
            
            num_articles = random.randint(5, 25)
            
            return {
                'score': max(-1.0, min(1.0, base_sentiment)),
                'confidence': min(0.9, 0.3 + (num_articles / 30)),
                'articles_analyzed': num_articles,
                'avg_sentiment': base_sentiment,
                'source': 'financial_news'
            }
            
        except Exception as e:
            return {
                'error': f'News sentiment analysis failed: {str(e)}',
                'score': 0.0,
                'confidence': 0.0
            }
    
    def _analyze_social_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze sentiment from social media (simulated for demo)"""
        try:
            # In a real implementation, this would:
            # 1. Fetch tweets, Reddit posts, etc. about the ticker
            # 2. Use sentiment analysis models
            # 3. Handle noise and spam filtering
            
            # Simulate social media sentiment (tends to be more volatile)
            base_sentiment = random.uniform(-0.6, 0.6)
            
            # Social media tends to be more extreme
            if abs(base_sentiment) < 0.2:
                base_sentiment *= 1.5  # Amplify mild sentiments
            
            num_posts = random.randint(50, 500)
            
            return {
                'score': max(-1.0, min(1.0, base_sentiment)),
                'confidence': min(0.7, 0.2 + (num_posts / 1000)),
                'posts_analyzed': num_posts,
                'avg_sentiment': base_sentiment,
                'source': 'social_media'
            }
            
        except Exception as e:
            return {
                'error': f'Social sentiment analysis failed: {str(e)}',
                'score': 0.0,
                'confidence': 0.0
            }
    
    def _analyze_analyst_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze sentiment from analyst reports and recommendations"""
        try:
            # In a real implementation, this would:
            # 1. Fetch analyst ratings and price targets
            # 2. Analyze recent upgrades/downgrades
            # 3. Weight by analyst reputation
            
            # Simulate analyst sentiment (tends to be more conservative)
            base_sentiment = random.uniform(-0.4, 0.4)
            
            # Analysts tend to be more moderate
            base_sentiment *= 0.8
            
            num_analysts = random.randint(3, 15)
            
            # Generate mock analyst actions
            recent_actions = []
            for _ in range(random.randint(0, 3)):
                action_type = random.choice(['upgrade', 'downgrade', 'maintain', 'initiate'])
                recent_actions.append(action_type)
            
            # Adjust sentiment based on recent actions
            upgrade_count = recent_actions.count('upgrade')
            downgrade_count = recent_actions.count('downgrade')
            
            if upgrade_count > downgrade_count:
                base_sentiment += 0.2
            elif downgrade_count > upgrade_count:
                base_sentiment -= 0.2
            
            return {
                'score': max(-1.0, min(1.0, base_sentiment)),
                'confidence': min(0.8, 0.4 + (num_analysts / 20)),
                'analysts_coverage': num_analysts,
                'recent_actions': recent_actions,
                'source': 'analyst_reports'
            }
            
        except Exception as e:
            return {
                'error': f'Analyst sentiment analysis failed: {str(e)}',
                'score': 0.0,
                'confidence': 0.0
            }
    
    def _calculate_sentiment_confidence(self, news: Dict, social: Dict, analyst: Dict) -> float:
        """Calculate overall confidence in sentiment analysis"""
        confidences = []
        
        for sentiment_data in [news, social, analyst]:
            if 'error' not in sentiment_data:
                confidences.append(sentiment_data.get('confidence', 0.0))
        
        if not confidences:
            return 0.0
        
        # Average confidence, but boost if multiple sources agree
        avg_confidence = sum(confidences) / len(confidences)
        
        # Check for agreement between sources
        scores = [s.get('score', 0) for s in [news, social, analyst] if 'error' not in s]
        if len(scores) >= 2:
            score_agreement = 1.0 - (max(scores) - min(scores)) / 2.0
            avg_confidence *= (0.7 + 0.3 * score_agreement)
        
        return min(1.0, avg_confidence)
    
    def _generate_sentiment_reasoning(self, news: Dict, social: Dict, 
                                    analyst: Dict, combined_score: float) -> str:
        """Generate human-readable reasoning for sentiment analysis"""
        reasoning_parts = []
        
        # Overall sentiment assessment
        if combined_score > 0.3:
            reasoning_parts.append("Overall market sentiment is positive.")
        elif combined_score < -0.3:
            reasoning_parts.append("Overall market sentiment is negative.")
        else:
            reasoning_parts.append("Market sentiment is mixed or neutral.")
        
        # Breakdown of sources
        for source, data in zip(['News', 'Social Media', 'Analyst Reports'], [news, social, analyst]):
            if 'error' not in data:
                reasoning_parts.append(f"{source} sentiment score: {data['score']:.2f} (confidence: {data['confidence']:.2f})")
                if 'recent_actions' in data:
                    actions = data['recent_actions']
                    reasoning_parts.append(f"Recent analyst actions: {', '.join(actions)}")
                if 'articles_analyzed' in data:
                    reasoning_parts.append(f"Articles analyzed: {data['articles_analyzed']}")
        
        # Confidence in overall sentiment
        reasoning_parts.append(f"Overall sentiment confidence: {combined_score:.2f}")
        
        return " | ".join(reasoning_parts)
    
    def _simulate_sentiment_analysis(self, ticker: str) -> Dict[str, Any]:
        """Simulate sentiment analysis when real data is not available"""
        # Generate simulated sentiment with some realism
        base_sentiment = random.uniform(0.3, 0.7)  # Generally neutral to positive
        
        # Add ticker-specific tendencies
        if ticker in ['AAPL', 'MSFT', 'GOOGL']:
            base_sentiment += 0.1  # Large caps tend to have positive sentiment
        elif ticker in ['TSLA']:
            base_sentiment += random.uniform(-0.2, 0.3)  # More volatile
        
        confidence = random.uniform(0.6, 0.9)
        num_sources = random.randint(8, 20)
        
        return {
            'score': base_sentiment,
            'confidence': confidence,
            'reasoning': f'Simulated analysis of {num_sources} sources shows {self._sentiment_label(base_sentiment)} sentiment',
            'articles_analyzed': num_sources,
            'source': 'simulated',
            'sentiment_distribution': {
                'positive': random.randint(3, 8),
                'neutral': random.randint(2, 6),
                'negative': random.randint(1, 4)
            }
        }
    
    def _sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.7:
            return "very positive"
        elif score > 0.6:
            return "positive"
        elif score > 0.4:
            return "neutral"
        elif score > 0.3:
            return "negative"
        else:
            return "very negative"
