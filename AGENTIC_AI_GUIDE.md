# Agentic Finance Bot - LLM Integration Guide

## Overview
The Agentic Finance Bot includes optional LLM (Large Language Model) integration for enhanced natural language explanations and insights. The system works in three modes:

1. **Simulation Mode** (Default) - Uses intelligent templates and simulated responses
2. **Real LLM Mode** - Integrates with actual LLM APIs (OpenAI, Anthropic, etc.)
3. **Local LLM Mode** - Uses locally hosted models (Ollama, etc.)

## Current Status
✅ **Agentic System**: Fully implemented with 6 specialized agents
✅ **LLM Agent**: Implemented with simulation and real API support
✅ **Demo**: Complete demonstration of all capabilities
✅ **API Integration**: Ready for OpenAI, Anthropic, and local models

## Quick Start with Agentic System

### Basic Usage
```python
from agentic_portfolio_manager import AgenticPortfolioManager

# Initialize with default settings (simulation mode)
manager = AgenticPortfolioManager(use_ml=True, use_llm=True)

# Analyze a stock with all agents
result = manager.analyze_stock("AAPL")
print(f"Recommendation: {result['recommendation']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Agents Used: {result['agents_used']}")
```

### Advanced Analysis
```python
# Get detailed agent-by-agent breakdown
detailed = manager.get_detailed_analysis("AAPL")
for agent, info in detailed['by_agent'].items():
    print(f"{agent}: {info['score']:.2f} ({info['confidence']:.1%})")

# Compare multiple stocks
comparison = manager.batch_analyze(["AAPL", "MSFT", "GOOGL"])
best_stock = max(comparison.items(), key=lambda x: x[1]['composite_score'])
print(f"Best opportunity: {best_stock[0]}")
```

## Agent Capabilities

### 1. Technical Analysis Agent
- RSI, MACD, Bollinger Bands, Moving Averages
- Price action and trend analysis
- Support/resistance levels

### 2. Sentiment Analysis Agent
- News sentiment analysis (with fallback simulation)
- Social media sentiment (simulated)
- Analyst sentiment tracking

### 3. Fundamental Analysis Agent
- Financial metrics simulation
- Valuation ratios
- Company health assessment

### 4. ML Prediction Agent
- Ensemble machine learning models
- Price movement prediction
- Feature engineering

### 5. Regime Detection Agent
- Market regime classification
- Volatility analysis
- Trend strength measurement

### 6. LLM Explanation Agent
- Natural language explanations
- Multi-agent synthesis
- Investment thesis generation

## Enabling Real LLM Integration

### Option 1: OpenAI Integration

1. **Install OpenAI package:**
```bash
pip install openai
```

2. **Get API key from OpenAI:**
   - Visit https://platform.openai.com/api-keys
   - Create a new API key
   - Set environment variable: `export OPENAI_API_KEY="your-key"`

3. **Enable in code:**
```python
from agentic_portfolio_manager import AgenticPortfolioManager

manager = AgenticPortfolioManager(use_llm=True)

# Enable real LLM with your API key
manager.agent_coordinator.agents['LLMExplanation'].enable_real_llm(
    api_key="your-openai-api-key",
    provider="openai"
)

# Now analyses will use GPT-4 for explanations
result = manager.analyze_stock("AAPL")
print(result['agentic_analysis']['agent_results']['LLMExplanation'])
```

### Option 2: Anthropic Claude Integration

1. **Install Anthropic package:**
```bash
pip install anthropic
```

2. **Get API key from Anthropic:**
   - Visit https://console.anthropic.com/
   - Create API key
   - Set environment variable: `export ANTHROPIC_API_KEY="your-key"`

3. **Enable in code:**
```python
manager.agent_coordinator.agents['LLMExplanation'].enable_real_llm(
    api_key="your-anthropic-api-key",
    provider="anthropic"
)
```

### Option 3: Local LLM Integration (Ollama)

1. **Install Ollama:**
```bash
# macOS
brew install ollama

# Start Ollama service
ollama serve

# Pull a model (e.g., Llama 2)
ollama pull llama2
```

2. **Enable in code:**
```python
manager.agent_coordinator.agents['LLMExplanation'].enable_real_llm(
    api_key="local",  # Special key for local models
    provider="local"
)
```

## Demo and Testing

### Run Complete Demo
```bash
cd /Users/tanis/Documents/FinanceBot
python advanced_demo.py
```

### Test Individual Components
```python
# Test individual agents
from agents import TechnicalAnalysisAgent, SentimentAnalysisAgent

tech_agent = TechnicalAnalysisAgent()
result = tech_agent.analyze("AAPL")
print(f"Technical score: {result['score']}")

sentiment_agent = SentimentAnalysisAgent()
result = sentiment_agent.analyze("AAPL") 
print(f"Sentiment score: {result['score']}")
```

### Test LLM Integration
```python
from agents import LLMExplanationAgent

llm_agent = LLMExplanationAgent()

# Test simulation mode
result = llm_agent.analyze("AAPL", {
    "TechnicalAnalysis": {"score": 0.7, "confidence": 0.8, "signal": "BUY"},
    "SentimentAnalysis": {"score": 0.6, "confidence": 0.9, "signal": "BUY"}
})
print("Simulation result:", result['reasoning'])

# Test real LLM (if API key provided)
llm_agent.enable_real_llm("your-api-key", "openai")
result = llm_agent.analyze("AAPL", {
    "TechnicalAnalysis": {"score": 0.7, "confidence": 0.8, "signal": "BUY"},
    "SentimentAnalysis": {"score": 0.6, "confidence": 0.9, "signal": "BUY"}
})
print("Real LLM result:", result['detailed_analysis']['full_text'])
```

## Configuration Options

### Agent Configuration
```python
# Disable/enable specific agents
manager.disable_agent("MLPrediction")  # Disable ML agent
manager.enable_agent("MLPrediction")   # Re-enable ML agent

# Get agent status
status = manager.get_agent_status()
print("Agent status:", status)

# Get performance stats
perf = manager.get_agent_performance()
print("Performance:", perf)
```

### Parallel vs Sequential Execution
```python
# Parallel execution (faster)
manager = AgenticPortfolioManager(parallel_execution=True)

# Sequential execution (more predictable)
manager = AgenticPortfolioManager(parallel_execution=False)
```

## API Rate Limits and Costs

### OpenAI GPT-4
- **Rate Limit**: 500 requests/minute
- **Cost**: ~$0.03/1K tokens (input), ~$0.06/1K tokens (output)
- **Typical Analysis**: ~500-1000 tokens = $0.02-0.05 per analysis

### Anthropic Claude
- **Rate Limit**: Varies by plan
- **Cost**: ~$0.015/1K tokens (input), ~$0.075/1K tokens (output)
- **Typical Analysis**: ~500-1000 tokens = $0.01-0.08 per analysis

### Local Models (Free)
- **Rate Limit**: Hardware dependent
- **Cost**: Free after initial setup
- **Performance**: Varies by model size and hardware

## Troubleshooting

### Common Issues

1. **"AgenticPortfolioManager not found"**
   - Ensure you're in the correct directory
   - Check that all agent files are present

2. **"LLM API error"**
   - Verify API key is correct
   - Check rate limits
   - System falls back to simulation mode automatically

3. **"Insufficient data"**
   - Some agents need minimum data points
   - Try different ticker symbols
   - Check internet connection for data collection

4. **"ML models not found"**
   - Run `manager.train_ml_models("AAPL")` first
   - ML agent will use simulation if models unavailable

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all agent activities will be logged
manager = AgenticPortfolioManager()
```

## Production Considerations

### For Production Use:
1. **Implement proper API key management** (environment variables, secrets manager)
2. **Add rate limiting and retry logic** for API calls
3. **Cache results** to avoid redundant API calls
4. **Monitor costs** for LLM API usage
5. **Implement proper error handling** and fallbacks
6. **Add data validation** for all inputs
7. **Consider model fine-tuning** for domain-specific performance

### Security Notes:
- Never hardcode API keys in source code
- Use environment variables or secure key management
- Implement proper authentication for production APIs
- Monitor API usage and set spending limits

## Future Enhancements

The system is designed to be easily extensible:

- **New Agents**: Add specialized agents for options, crypto, etc.
- **Enhanced LLM Integration**: Fine-tuned models for financial analysis
- **Real-time Data**: Streaming market data integration
- **Portfolio Optimization**: Advanced portfolio construction algorithms
- **Risk Management**: More sophisticated risk assessment
- **Backtesting**: Historical strategy validation
- **Web Interface**: Dashboard for interactive analysis

## Support and Documentation

- **Code**: All agents are well-documented with docstrings
- **Examples**: See `advanced_demo.py` for comprehensive examples
- **Logs**: Enable debug logging for troubleshooting
- **Performance**: Monitor agent execution times and success rates

The agentic system provides a robust, scalable foundation for AI-powered financial analysis with optional LLM enhancement for natural language insights.
