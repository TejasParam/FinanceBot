"""
Agentic AI module for finance analysis.

This module contains specialized agents for different aspects of financial analysis:
- Technical Analysis Agent
- Sentiment Analysis Agent  
- ML Prediction Agent
- Regime Detection Agent
- Fundamental Analysis Agent
- LLM Explanation Agent
- Agent Coordinator (orchestrates all agents)
"""

from .base_agent import BaseAgent
from .technical_agent import TechnicalAnalysisAgent
from .sentiment_agent import SentimentAnalysisAgent
from .ml_agent import MLPredictionAgent
from .regime_agent import RegimeDetectionAgent
from .fundamental_agent import FundamentalAnalysisAgent
from .llm_agent import LLMExplanationAgent
from .coordinator import AgentCoordinator

__all__ = [
    'BaseAgent',
    'TechnicalAnalysisAgent',
    'SentimentAnalysisAgent', 
    'MLPredictionAgent',
    'RegimeDetectionAgent',
    'FundamentalAnalysisAgent',
    'LLMExplanationAgent',
    'AgentCoordinator'
]
