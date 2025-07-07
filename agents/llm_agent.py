"""
LLM Explanation Agent for generating natural language explanations and insights.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Optional
import json
import random
from .base_agent import BaseAgent

class LLMExplanationAgent(BaseAgent):
    """
    Agent specialized in generating natural language explanations and insights.
    Uses LLM-like reasoning to synthesize analysis from other agents into 
    coherent, actionable explanations.
    
    Note: This is a simulated LLM agent for demo purposes. In production,
    you would integrate with actual LLM APIs like OpenAI, Anthropic, etc.
    """
    
    def __init__(self, use_real_llm: bool = False, api_key: Optional[str] = None):
        super().__init__("LLMExplanation")
        self.use_real_llm = use_real_llm
        self.api_key = api_key
        
        # Template responses for simulation
        self.explanation_templates = {
            'bullish': [
                "The analysis reveals strong bullish indicators for {ticker}. {key_factors} These factors suggest potential upward momentum in the near term.",
                "Multiple positive signals align for {ticker}. {key_factors} This convergence of favorable conditions supports a bullish outlook.",
                "The data points to attractive opportunities in {ticker}. {key_factors} These strengths position the stock well for potential gains."
            ],
            'bearish': [
                "Several concerning factors emerge in the {ticker} analysis. {key_factors} These issues suggest caution and potential downside risk.",
                "The analysis reveals challenging conditions for {ticker}. {key_factors} These headwinds may pressure the stock in the coming period.",
                "Warning signs are evident in {ticker}'s current setup. {key_factors} Investors should be aware of these risk factors."
            ],
            'neutral': [
                "The analysis shows mixed signals for {ticker}. {key_factors} This suggests a wait-and-see approach may be prudent.",
                "Balanced conditions emerge in the {ticker} analysis. {key_factors} The stock appears fairly valued with limited immediate catalysts.",
                "Competing factors balance out in {ticker}'s current situation. {key_factors} This creates an uncertain but stable outlook."
            ]
        }
        
    def analyze(self, ticker: str, agent_results: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate comprehensive explanation by synthesizing results from all agents.
        
        Args:
            ticker: Stock ticker symbol
            agent_results: Dictionary containing results from other agents
            
        Returns:
            Dictionary with LLM-generated explanations and insights
        """
        try:
            if self.use_real_llm and self.api_key:
                return self._generate_real_llm_explanation(ticker, agent_results)
            else:
                return self._generate_simulated_explanation(ticker, agent_results)
                
        except Exception as e:
            return {
                'error': f'LLM explanation failed: {str(e)}',
                'score': 0.0,
                'confidence': 0.0,
                'reasoning': f'LLM explanation error: {str(e)[:100]}'
            }
    
    def _generate_simulated_explanation(self, ticker: str, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate simulated LLM-style explanation"""
        # Aggregate scores from all agents
        agent_scores = []
        agent_insights = []
        
        for agent_name, result in agent_results.items():
            if isinstance(result, dict) and 'error' not in result:
                score = result.get('score', 0.0)
                confidence = result.get('confidence', 0.5)
                reasoning = result.get('reasoning', '')
                
                # Weight by confidence
                weighted_score = score * confidence
                agent_scores.append(weighted_score)
                
                # Extract key insights
                if reasoning:
                    agent_insights.append({
                        'agent': agent_name,
                        'score': score,
                        'confidence': confidence,
                        'key_point': reasoning.split('.')[0] + '.' if '.' in reasoning else reasoning
                    })
        
        # Calculate overall sentiment
        overall_score = sum(agent_scores) / len(agent_scores) if agent_scores else 0.0
        overall_confidence = self._calculate_explanation_confidence(agent_results)
        
        # Determine sentiment category
        if overall_score > 0.2:
            sentiment = 'bullish'
        elif overall_score < -0.2:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        # Generate key factors summary
        key_factors = self._synthesize_key_factors(agent_insights, sentiment)
        
        # Select and customize explanation template
        template = random.choice(self.explanation_templates[sentiment])
        main_explanation = template.format(ticker=ticker, key_factors=key_factors)
        
        # Generate additional insights
        market_context = self._generate_market_context(agent_results)
        risk_factors = self._identify_risk_factors(agent_results)
        opportunities = self._identify_opportunities(agent_results)
        
        # Generate action recommendations
        recommendations = self._generate_recommendations(overall_score, overall_confidence, agent_results)
        
        return {
            'score': overall_score,
            'confidence': overall_confidence,
            'reasoning': main_explanation,
            'sentiment': sentiment,
            'detailed_analysis': {
                'main_explanation': main_explanation,
                'market_context': market_context,
                'key_strengths': opportunities,
                'key_risks': risk_factors,
                'agent_consensus': self._analyze_agent_consensus(agent_results)
            },
            'recommendations': recommendations,
            'summary': self._generate_executive_summary(ticker, overall_score, sentiment),
            'agents_analyzed': len([r for r in agent_results.values() if isinstance(r, dict) and 'error' not in r])
        }
    
    def _synthesize_key_factors(self, agent_insights: List[Dict], sentiment: str) -> str:
        """Synthesize key factors from agent insights"""
        if not agent_insights:
            return "Limited data available for comprehensive analysis."
        
        # Group insights by positive/negative sentiment
        positive_insights = [i for i in agent_insights if i['score'] > 0.1]
        negative_insights = [i for i in agent_insights if i['score'] < -0.1]
        
        factors = []
        
        if sentiment == 'bullish' and positive_insights:
            # Focus on positive factors
            top_positive = sorted(positive_insights, key=lambda x: x['score'] * x['confidence'], reverse=True)[:2]
            for insight in top_positive:
                factors.append(f"{insight['agent'].replace('_', ' ').title()} shows {insight['key_point'].lower()}")
                
        elif sentiment == 'bearish' and negative_insights:
            # Focus on negative factors
            top_negative = sorted(negative_insights, key=lambda x: abs(x['score']) * x['confidence'], reverse=True)[:2]
            for insight in top_negative:
                factors.append(f"{insight['agent'].replace('_', ' ').title()} indicates {insight['key_point'].lower()}")
                
        else:
            # Mixed signals - show both sides
            if positive_insights:
                best_positive = max(positive_insights, key=lambda x: x['score'] * x['confidence'])
                factors.append(f"Positive: {best_positive['key_point'].lower()}")
            if negative_insights:
                best_negative = max(negative_insights, key=lambda x: abs(x['score']) * x['confidence'])
                factors.append(f"Negative: {best_negative['key_point'].lower()}")
        
        return " ".join(factors) if factors else "Analysis shows mixed technical and fundamental signals."
    
    def _generate_market_context(self, agent_results: Dict[str, Any]) -> str:
        """Generate market context from regime and sentiment analysis"""
        context_parts = []
        
        # Regime analysis context
        if 'RegimeDetection' in agent_results:
            regime_result = agent_results['RegimeDetection']
            if 'error' not in regime_result:
                regime = regime_result.get('regime', 'unknown')
                context_parts.append(f"Current market regime appears to be {regime.replace('_', ' ')}")
        
        # Sentiment context
        if 'SentimentAnalysis' in agent_results:
            sentiment_result = agent_results['SentimentAnalysis']
            if 'error' not in sentiment_result:
                score = sentiment_result.get('score', 0)
                if score > 0.2:
                    context_parts.append("with positive market sentiment")
                elif score < -0.2:
                    context_parts.append("with negative market sentiment")
                else:
                    context_parts.append("with neutral market sentiment")
        
        return ". ".join(context_parts) + "." if context_parts else "Market context analysis unavailable."
    
    def _identify_risk_factors(self, agent_results: Dict[str, Any]) -> List[str]:
        """Identify key risk factors from agent analyses"""
        risks = []
        
        for agent_name, result in agent_results.items():
            if isinstance(result, dict) and 'error' not in result:
                score = result.get('score', 0.0)
                confidence = result.get('confidence', 0.5)
                
                # Identify significant negative signals
                if score < -0.3 and confidence > 0.6:
                    if agent_name == 'TechnicalAnalysis':
                        risks.append("Technical indicators show bearish patterns")
                    elif agent_name == 'FundamentalAnalysis':
                        risks.append("Fundamental metrics indicate overvaluation or weak financials")
                    elif agent_name == 'SentimentAnalysis':
                        risks.append("Market sentiment has turned negative")
                    elif agent_name == 'RegimeDetection':
                        risks.append("Current market regime suggests higher volatility or downtrend")
                    elif agent_name == 'MLPrediction':
                        risks.append("Machine learning models predict downward pressure")
        
        return risks[:3]  # Limit to top 3 risks
    
    def _identify_opportunities(self, agent_results: Dict[str, Any]) -> List[str]:
        """Identify key opportunities from agent analyses"""
        opportunities = []
        
        for agent_name, result in agent_results.items():
            if isinstance(result, dict) and 'error' not in result:
                score = result.get('score', 0.0)
                confidence = result.get('confidence', 0.5)
                
                # Identify significant positive signals
                if score > 0.3 and confidence > 0.6:
                    if agent_name == 'TechnicalAnalysis':
                        opportunities.append("Technical setup shows bullish momentum")
                    elif agent_name == 'FundamentalAnalysis':
                        opportunities.append("Strong fundamental metrics support higher valuation")
                    elif agent_name == 'SentimentAnalysis':
                        opportunities.append("Positive market sentiment provides tailwind")
                    elif agent_name == 'RegimeDetection':
                        opportunities.append("Favorable market regime supports upward moves")
                    elif agent_name == 'MLPrediction':
                        opportunities.append("ML models predict positive price movement")
        
        return opportunities[:3]  # Limit to top 3 opportunities
    
    def _analyze_agent_consensus(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consensus among agents"""
        scores = []
        confidences = []
        
        for result in agent_results.values():
            if isinstance(result, dict) and 'error' not in result:
                scores.append(result.get('score', 0.0))
                confidences.append(result.get('confidence', 0.5))
        
        if not scores:
            return {'consensus': 'unknown', 'strength': 0.0}
        
        # Calculate consensus metrics
        avg_score = sum(scores) / len(scores)
        score_std = (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5
        avg_confidence = sum(confidences) / len(confidences)
        
        # Determine consensus strength (lower std dev = higher consensus)
        consensus_strength = max(0, 1 - score_std)
        
        # Determine consensus direction
        if avg_score > 0.2:
            consensus = 'bullish'
        elif avg_score < -0.2:
            consensus = 'bearish'
        else:
            consensus = 'neutral'
        
        return {
            'consensus': consensus,
            'strength': consensus_strength,
            'average_score': avg_score,
            'average_confidence': avg_confidence,
            'agreement_level': 'high' if consensus_strength > 0.7 else 'medium' if consensus_strength > 0.4 else 'low'
        }
    
    def _generate_recommendations(self, score: float, confidence: float, agent_results: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Primary recommendation based on score and confidence
        if score > 0.3 and confidence > 0.6:
            recommendations.append("Consider buying on current strength or on any minor pullbacks")
        elif score > 0.1 and confidence > 0.5:
            recommendations.append("Consider small position or wait for better entry point")
        elif score < -0.3 and confidence > 0.6:
            recommendations.append("Consider selling or avoiding new positions")
        elif score < -0.1 and confidence > 0.5:
            recommendations.append("Exercise caution and consider reducing position size")
        else:
            recommendations.append("Hold current position and monitor for clearer signals")
        
        # Risk management
        if confidence < 0.5:
            recommendations.append("Use smaller position sizes due to signal uncertainty")
        
        # Timing considerations
        regime_result = agent_results.get('RegimeDetection', {})
        if 'error' not in regime_result:
            regime = regime_result.get('regime', '')
            if 'high_volatility' in regime:
                recommendations.append("Consider using stop-losses due to high volatility environment")
        
        return recommendations
    
    def _generate_executive_summary(self, ticker: str, score: float, sentiment: str) -> str:
        """Generate a concise executive summary"""
        action = "BUY" if score > 0.3 else "SELL" if score < -0.3 else "HOLD"
        
        summaries = {
            'bullish': f"{ticker} shows {sentiment} signals. Recommendation: {action}. Multiple factors align positively.",
            'bearish': f"{ticker} shows {sentiment} signals. Recommendation: {action}. Several risk factors identified.",
            'neutral': f"{ticker} shows {sentiment} signals. Recommendation: {action}. Mixed signals suggest patience."
        }
        
        return summaries.get(sentiment, f"{ticker} analysis complete. Recommendation: {action}.")
    
    def _calculate_explanation_confidence(self, agent_results: Dict[str, Any]) -> float:
        """Calculate confidence in the overall explanation"""
        confidences = []
        
        for result in agent_results.values():
            if isinstance(result, dict) and 'error' not in result:
                confidences.append(result.get('confidence', 0.5))
        
        if not confidences:
            return 0.0
        
        # Average confidence with bonus for having multiple agents
        avg_confidence = sum(confidences) / len(confidences)
        agent_bonus = min(0.2, len(confidences) * 0.05)  # Up to 20% bonus for 4+ agents
        
        return min(1.0, avg_confidence + agent_bonus)
    
    def _generate_real_llm_explanation(self, ticker: str, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation using real LLM API"""
        
        # Prepare comprehensive prompt for LLM
        prompt = self._create_comprehensive_prompt(ticker, agent_results)
        
        try:
            # Check if using local LLM
            if self.api_key == "local":
                return self._call_local_llm(prompt, ticker, agent_results)
            
            # Try OpenAI
            elif self.api_key and 'sk-' in str(self.api_key):
                return self._call_openai_api(prompt, ticker, agent_results)
            
            # Try Anthropic Claude
            elif self.api_key and 'anthropic' in str(self.api_key).lower():
                return self._call_anthropic_api(prompt, ticker, agent_results)
            
            # Default to local LLM if available
            else:
                return self._call_local_llm(prompt, ticker, agent_results)
                
        except Exception as e:
            print(f"LLM API call failed: {e}")
            return self._generate_simulated_explanation(ticker, agent_results)
    
    def _create_comprehensive_prompt(self, ticker: str, agent_results: Dict[str, Any]) -> str:
        """Create comprehensive analysis prompt for LLM"""
        
        # Extract key data from each agent
        agent_summaries = []
        for agent_name, result in agent_results.items():
            if isinstance(result, dict) and 'error' not in result:
                summary = {
                    'agent': agent_name,
                    'score': result.get('score', 0.0),
                    'confidence': result.get('confidence', 0.0),
                    'signal': result.get('signal', 'HOLD'),
                    'reasoning': result.get('reasoning', '')[:200]  # Limit length
                }
                agent_summaries.append(summary)
        
        prompt = f"""
You are a senior financial analyst providing investment insights for {ticker}.

AGENT ANALYSIS RESULTS:
{chr(10).join([f"- {a['agent']}: Score={a['score']:.2f}, Signal={a['signal']}, Confidence={a['confidence']:.1%}, Reasoning: {a['reasoning']}" for a in agent_summaries])}

Based on this comprehensive multi-agent analysis, provide:

1. EXECUTIVE SUMMARY (2-3 sentences)
   - Clear investment recommendation (BUY/SELL/HOLD)
   - Primary reasoning for recommendation
   - Overall confidence level

2. KEY INSIGHTS
   - Most important bullish factors
   - Most important bearish factors
   - Critical uncertainties or risks

3. INVESTMENT THESIS
   - Why this is a good/bad investment opportunity
   - Time horizon considerations
   - Risk-reward assessment

4. ACTIONABLE RECOMMENDATIONS
   - Specific entry/exit strategies
   - Position sizing considerations
   - Key metrics to monitor

Keep your analysis concise, actionable, and focused on synthesizing the agent insights into coherent investment guidance.
"""
        return prompt
    
    def _call_openai_api(self, prompt: str, ticker: str, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Call OpenAI API for LLM analysis"""
        try:
            import openai
            
            # Configure OpenAI
            openai.api_key = self.api_key
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a senior financial analyst specializing in investment research and portfolio management."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content
            
            # Parse LLM response into structured format
            return self._parse_llm_response(analysis_text, ticker, agent_results)
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
    
    def _call_anthropic_api(self, prompt: str, ticker: str, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Call Anthropic Claude API for LLM analysis"""
        try:
            import anthropic
            
            client = anthropic.Client(api_key=self.api_key)
            
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            analysis_text = response.content[0].text
            
            # Parse LLM response into structured format
            return self._parse_llm_response(analysis_text, ticker, agent_results)
            
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}")
    
    def _call_local_llm(self, prompt: str, ticker: str, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Call local LLM model (e.g., Ollama, local transformers)"""
        try:
            # Try Ollama first (preferred local LLM)
            if self._test_ollama_connection():
                return self._call_ollama_api(prompt, ticker, agent_results)
            
            # Fallback to other local models if available
            # You can add other local LLM integrations here
            else:
                raise Exception("No local LLM service available")
                
        except Exception as e:
            raise Exception(f"Local LLM error: {e}")
    
    def _test_ollama_connection(self) -> bool:
        """Test if Ollama service is running and has models available"""
        try:
            import requests
            
            # Check if Ollama is running
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                # Check if we have any models available
                return len(models) > 0
            return False
        except:
            return False
    
    def _call_ollama_api(self, prompt: str, ticker: str, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Call Ollama local API for LLM analysis"""
        import requests
        
        # Get available models
        models_response = requests.get('http://localhost:11434/api/tags', timeout=5)
        available_models = [m['name'] for m in models_response.json().get('models', [])]
        
        # Choose best available model (prefer newer/larger models)
        preferred_models = ['llama3.2:3b', 'llama3.1', 'llama3', 'llama2', 'mistral', 'codellama']
        model_to_use = None
        
        for preferred in preferred_models:
            for available in available_models:
                if preferred in available:
                    model_to_use = available
                    break
            if model_to_use:
                break
        
        if not model_to_use:
            model_to_use = available_models[0] if available_models else 'llama2'
        
        print(f"🤖 Using local LLM: {model_to_use}")
        
        # Make the API call
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model_to_use,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.3,
                    'num_predict': 800  # Limit response length
                }
            },
            timeout=60  # Increased timeout for local processing
        )
        
        if response.status_code == 200:
            analysis_text = response.json().get('response', '')
            print(f"✅ Local LLM analysis completed ({len(analysis_text)} chars)")
            return self._parse_llm_response(analysis_text, ticker, agent_results)
        else:
            raise Exception(f"Ollama API returned status {response.status_code}: {response.text}")
    
    def _parse_llm_response(self, analysis_text: str, ticker: str, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        
        # Extract overall recommendation and confidence
        recommendation = "HOLD"
        confidence = 0.7
        
        text_lower = analysis_text.lower()
        if any(word in text_lower for word in ['strong buy', 'buy', 'bullish']):
            recommendation = "BUY"
            confidence = 0.8
        elif any(word in text_lower for word in ['sell', 'bearish', 'avoid']):
            recommendation = "SELL"
            confidence = 0.8
        
        # Calculate composite score from agent results
        agent_scores = [r.get('score', 0.0) for r in agent_results.values() 
                       if isinstance(r, dict) and 'error' not in r]
        overall_score = sum(agent_scores) / len(agent_scores) if agent_scores else 0.5
        
        return {
            'score': overall_score,
            'confidence': confidence,
            'reasoning': analysis_text[:500],  # Truncate for storage
            'signal': recommendation,
            'detailed_analysis': {
                'full_text': analysis_text,
                'main_explanation': analysis_text.split('\n')[0] if analysis_text else '',
                'investment_thesis': self._extract_section(analysis_text, 'INVESTMENT THESIS'),
                'key_insights': self._extract_section(analysis_text, 'KEY INSIGHTS'),
                'recommendations': self._extract_section(analysis_text, 'ACTIONABLE RECOMMENDATIONS'),
                'llm_model': 'real_api'  # Mark this as real LLM output
            },
            'llm_model': 'real_api',
            'agent_synthesis': True,
            'local_llm_used': self.api_key == "local"
        }
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract specific section from LLM response"""
        lines = text.split('\n')
        in_section = False
        section_lines = []
        
        for line in lines:
            if section_name.upper() in line.upper():
                in_section = True
                continue
            elif in_section and line.strip() and any(char.isdigit() for char in line[:5]):
                break  # Hit next numbered section
            elif in_section:
                section_lines.append(line.strip())
        
        return '\n'.join(section_lines).strip()
    
    def enable_real_llm(self, api_key: str, provider: str = "openai"):
        """Enable real LLM integration with API key"""
        self.api_key = api_key
        self.use_real_llm = True
        print(f"✅ Real LLM integration enabled with {provider}")
        
        # Test connection
        try:
            test_result = self.analyze("AAPL", {
                "TechnicalAnalysis": {"score": 0.7, "confidence": 0.8, "signal": "BUY", "reasoning": "Test"}
            })
            if 'error' not in test_result:
                print("✅ LLM API connection successful")
            else:
                print(f"⚠️ LLM API test failed: {test_result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"⚠️ LLM API test error: {e}")
    
    def enable_local_llm(self):
        """Enable local LLM integration (e.g., Ollama)"""
        self.use_real_llm = True
        self.api_key = "local"  # Use "local" as identifier for local LLM
        
        # Test Ollama connection
        if self._test_ollama_connection():
            print("✅ Local LLM (Ollama) integration enabled")
            
            # Test with a simple analysis
            try:
                test_result = self.analyze("AAPL", {
                    "TechnicalAnalysis": {"score": 0.7, "confidence": 0.8, "signal": "BUY", "reasoning": "Test"}
                })
                if 'error' not in test_result:
                    print("✅ Local LLM test analysis successful")
                else:
                    print(f"⚠️ Local LLM test failed: {test_result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"⚠️ Local LLM test error: {e}")
        else:
            print("❌ Local LLM not available. Make sure Ollama is running and has models installed.")
            print("   Start Ollama: brew services start ollama")
            print("   Install model: ollama pull llama3.2:3b")
            self.use_real_llm = False
            self.api_key = None
    
    def disable_real_llm(self):
        """Disable real LLM and use simulation"""
        self.use_real_llm = False
        self.api_key = None
        print("ℹ️ Real LLM disabled, using simulation mode")
