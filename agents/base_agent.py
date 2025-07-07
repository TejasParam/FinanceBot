"""
Base agent class for all financial analysis agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

class BaseAgent(ABC):
    """
    Abstract base class for all financial analysis agents.
    
    Each agent specializes in a specific domain (technical, sentiment, ML, etc.)
    and provides standardized analyze() method.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"Agent.{name}")
        self.enabled = True
        
    @abstractmethod
    def analyze(self, ticker: str, **kwargs) -> Dict[str, Any]:
        """
        Analyze the given ticker and return structured results.
        
        Args:
            ticker: Stock ticker symbol
            **kwargs: Additional parameters specific to the agent
            
        Returns:
            Dictionary with analysis results including:
            - score: Numerical score (typically -1 to 1 or 0 to 1)
            - confidence: Confidence level (0 to 1)
            - reasoning: Human-readable explanation
            - raw_data: Optional raw analysis data
            - error: Error message if analysis failed
        """
        pass
    
    def is_enabled(self) -> bool:
        """Check if this agent is enabled"""
        return self.enabled
    
    def enable(self):
        """Enable this agent"""
        self.enabled = True
        
    def disable(self):
        """Disable this agent"""
        self.enabled = False
        
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'type': self.__class__.__name__
        }
    
    def safe_analyze(self, ticker: str, **kwargs) -> Dict[str, Any]:
        """
        Safe wrapper around analyze() that handles exceptions.
        """
        if not self.enabled:
            return {
                'error': f'Agent {self.name} is disabled',
                'score': 0.0,
                'confidence': 0.0,
                'reasoning': f'Agent {self.name} is currently disabled'
            }
        
        try:
            result = self.analyze(ticker, **kwargs)
            # Ensure required fields are present
            if 'score' not in result:
                result['score'] = 0.0
            if 'confidence' not in result:
                result['confidence'] = 0.0
            if 'reasoning' not in result:
                result['reasoning'] = f'Analysis by {self.name}'
            return result
        except Exception as e:
            self.logger.error(f"Error in {self.name} analysis for {ticker}: {e}")
            return {
                'error': str(e),
                'score': 0.0,
                'confidence': 0.0,
                'reasoning': f'Error in {self.name} analysis: {str(e)[:100]}'
            }
