import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_collection import DataCollectionAgent
from technical_analysis import technical_analyst_agent

class risk_manager_agent:
    def __init__(self):
        self.data_collector = DataCollectionAgent()
        self.tech_analyst = technical_analyst_agent()
        
    def calculate_beta(self, ticker, market_ticker="^GSPC", period=1):
        """
        Calculate beta (market risk) for a stock
        Beta > 1: More volatile than market
        Beta < 1: Less volatile than market
        """
        try:
            print(f"Calculating beta for {ticker} relative to {market_ticker}...")
            
            # Use a simple approach: get 1 year of data with yfinance period parameter
            stock_ticker = yf.Ticker(ticker)
            market_ticker_obj = yf.Ticker(market_ticker)
            
            # Get historical data
            stock_hist = stock_ticker.history(period="1y")
            market_hist = market_ticker_obj.history(period="1y")
            
            if stock_hist.empty or market_hist.empty or len(stock_hist) < 50 or len(market_hist) < 50:
                print(f"Insufficient data for beta calculation")
                return None
            
            # Extract closing prices as simple lists
            stock_prices = stock_hist['Close'].tolist()
            market_prices = market_hist['Close'].tolist()
            
            # Calculate returns manually
            stock_returns = []
            market_returns = []
            
            # Only use the minimum length to ensure alignment
            min_length = min(len(stock_prices), len(market_prices))
            
            for i in range(1, min_length):
                stock_ret = (stock_prices[i] - stock_prices[i-1]) / stock_prices[i-1]
                market_ret = (market_prices[i] - market_prices[i-1]) / market_prices[i-1]
                stock_returns.append(stock_ret)
                market_returns.append(market_ret)
            
            if len(stock_returns) < 30:
                print(f"Insufficient return data for beta calculation")
                return None
            
            # Convert to numpy arrays for calculation
            stock_array = np.array(stock_returns)
            market_array = np.array(market_returns)
            
            # Calculate beta using simple statistics
            # Beta = Covariance(stock, market) / Variance(market)
            stock_mean = np.mean(stock_array)
            market_mean = np.mean(market_array)
            
            covariance = np.mean((stock_array - stock_mean) * (market_array - market_mean))
            market_variance = np.mean((market_array - market_mean) ** 2)
            
            if market_variance == 0:
                print("Market variance is zero, cannot calculate beta")
                return None
            
            beta = covariance / market_variance
            
            # Sanity check
            if np.isnan(beta) or beta < -3 or beta > 5:
                print(f"Beta value {beta} is outside reasonable range, using default")
                return 1.0
            
            return beta
            
        except Exception as e:
            print(f"Error calculating beta: {e}")
            return None
    
    def calculate_volatility(self, ticker, period=30, annualize=True):
        """Calculate historical volatility (standard deviation of returns)"""
        try:
            data = self.tech_analyst.getData(ticker)[0]  # Get only the historical data
            returns = data['Close'].pct_change().dropna().tail(period)
            vol = returns.std()
            
            # Annualize if requested (√252 is standard for daily data)
            if annualize:
                vol = vol * np.sqrt(252)
                
            return vol
        except Exception as e:
            print(f"Error calculating volatility: {e}")
            return None
    
    def calculate_var(self, ticker, confidence=0.95, period=30):
        """
        Calculate Value at Risk (VaR) using historical method
        Returns the amount that could be lost with given confidence level
        """
        try:
            data = self.tech_analyst.getData(ticker)[0]  # Get only the historical data
            returns = data['Close'].pct_change().dropna().tail(period)
            
            # Historical VaR calculation
            var = np.percentile(returns, 100 * (1 - confidence))
            
            # Convert to percentage
            var_pct = var * 100
            
            return abs(var_pct)  # Return as positive number for readability
        except Exception as e:
            print(f"Error calculating VaR: {e}")
            return None
    
    def calculate_max_drawdown(self, ticker, period=365):
        """Calculate maximum historical drawdown"""
        try:
            data = self.tech_analyst.getData(ticker)[0].tail(period)  # Get only the historical data
            price = data['Close']
            
            # Calculate running maximum
            running_max = price.cummax()
            
            # Calculate drawdown
            drawdown = (price / running_max - 1) * 100
            
            # Get the maximum drawdown
            max_drawdown = abs(drawdown.min())
            
            return max_drawdown
        except Exception as e:
            print(f"Error calculating max drawdown: {e}")
            return None
    
    def get_debt_to_equity(self, ticker):
        """Get debt-to-equity ratio (financial leverage risk)"""
        try:
            stock_info = yf.Ticker(ticker).info
            return stock_info.get('debtToEquity', None)
        except Exception as e:
            print(f"Error calculating debt-to-equity: {e}")
            return None
    
    def get_current_ratio(self, ticker):
        """Get current ratio (liquidity risk)"""
        try:
            stock_info = yf.Ticker(ticker).info
            return stock_info.get('currentRatio', None)
        except Exception as e:
            print(f"Error calculating current ratio: {e}")
            return None
    
    def calculate_liquidity_risk(self, ticker, period=30):
        """
        Calculate liquidity risk based on trading volume
        Returns liquidity score (higher means more liquid, less risk)
        """
        try:
            data = self.tech_analyst.getData(ticker)[0].tail(period)  # Get only the historical data
            
            # Average daily volume
            avg_volume = data['Volume'].mean()
            
            # Calculate volume volatility
            vol_volatility = data['Volume'].std() / avg_volume
            
            # Get market cap
            market_cap = yf.Ticker(ticker).info.get('marketCap', 1e9)
            
            # Calculate liquidity score (higher is better)
            # Based on dollar volume relative to market cap and volume consistency
            liquidity_score = (avg_volume * data['Close'].mean() / market_cap) * (1 / (1 + vol_volatility))
            
            # Scale from 0-10 (higher is more liquid, less risk)
            scaled_score = min(10, max(0, 10 * liquidity_score * 100))
            
            return scaled_score
        except Exception as e:
            print(f"Error calculating liquidity risk: {e}")
            return None
    
    def calculate_sharpe_ratio(self, ticker, risk_free_rate=0.05, period=252):
        """
        Calculate Sharpe ratio (risk-adjusted return)
        Higher is better: >1 is good, >2 is very good, >3 is excellent
        """
        try:
            data = self.tech_analyst.getData(ticker)[0].tail(period)  # Get only the historical data
            returns = data['Close'].pct_change().dropna()
            
            # Calculate annualized return and standard deviation
            avg_return = returns.mean() * 252  # Annualized return
            std_dev = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Calculate Sharpe ratio
            sharpe = (avg_return - risk_free_rate) / std_dev
            
            return sharpe
        except Exception as e:
            print(f"Error calculating Sharpe ratio: {e}")
            return None
    
    def calculate_sortino_ratio(self, ticker, risk_free_rate=0.05, period=252):
        """
        Calculate Sortino ratio (using only downside risk)
        Similar to Sharpe but only considers "bad" volatility
        """
        try:
            data = self.tech_analyst.getData(ticker)[0].tail(period)  # Get only the historical data
            returns = data['Close'].pct_change().dropna()
            
            # Calculate annualized return
            avg_return = returns.mean() * 252
            
            # Calculate downside deviation (only negative returns)
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252)
            
            # Handle case where there are no negative returns
            if downside_deviation == 0:
                return float('inf')  # Perfect Sortino ratio
            
            # Calculate Sortino ratio
            sortino = (avg_return - risk_free_rate) / downside_deviation
            
            return sortino
        except Exception as e:
            print(f"Error calculating Sortino ratio: {e}")
            return None
    
    def calculate_risk_score(self, ticker):
        """
        Calculate composite risk score from 1-10 (10 is highest risk)
        Combines multiple risk factors with weights
        """
        try:
            # Collect risk metrics
            risk_metrics = {}
            
            # Market risk (Beta)
            beta = self.calculate_beta(ticker)
            risk_metrics['beta'] = beta if beta is not None else 1.0
            
            # Volatility risk
            volatility = self.calculate_volatility(ticker)
            risk_metrics['volatility'] = volatility if volatility is not None else 0.3
            
            # Downside risk
            max_drawdown = self.calculate_max_drawdown(ticker)
            risk_metrics['max_drawdown'] = max_drawdown if max_drawdown is not None else 20.0
            
            # Financial risk
            debt_equity = self.get_debt_to_equity(ticker)
            risk_metrics['debt_equity'] = debt_equity if debt_equity is not None else 1.0
            
            # Liquidity risk
            liquidity = self.calculate_liquidity_risk(ticker)
            risk_metrics['liquidity'] = liquidity if liquidity is not None else 5.0
            
            # Risk-adjusted return
            sharpe = self.calculate_sharpe_ratio(ticker)
            risk_metrics['sharpe'] = sharpe if sharpe is not None else 1.0
            
            # Calculate weighted risk score
            risk_score = (
                0.20 * self.normalize_value(risk_metrics['beta'], 0.5, 2.0) +
                0.20 * self.normalize_value(risk_metrics['volatility'], 0.1, 0.5) +
                0.15 * self.normalize_value(risk_metrics['max_drawdown'], 10, 50) +
                0.15 * self.normalize_value(risk_metrics['debt_equity'], 0, 2) +
                0.15 * (1 - self.normalize_value(risk_metrics['liquidity'], 0, 10)) +
                0.15 * (1 - self.normalize_value(risk_metrics['sharpe'], 0, 3))
            )
            
            # Scale to 1-10
            scaled_score = min(10, max(1, risk_score * 10))
            
            return {
                'risk_score': scaled_score,
                'risk_metrics': risk_metrics,
                'risk_category': self.categorize_risk(scaled_score)
            }
        except Exception as e:
            print(f"Error calculating risk score: {e}")
            return {'risk_score': 5, 'risk_category': 'Moderate Risk (default)'}
    
    def normalize_value(self, value, min_val, max_val):
        """Normalize a value between 0 and 1"""
        if value is None:
            return 0.5  # Default if data is missing
        return min(1, max(0, (value - min_val) / (max_val - min_val)))
    
    def categorize_risk(self, risk_score):
        """Categorize risk score into descriptive category"""
        if risk_score < 3:
            return "Very Low Risk"
        elif risk_score < 5:
            return "Low Risk"
        elif risk_score < 7:
            return "Moderate Risk"
        elif risk_score < 9:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def recommend_position_size(self, ticker, portfolio_value, max_risk_pct=2):
        """
        Recommend position size based on risk assessment
        Returns dollar amount to invest and number of shares
        """
        try:
            # Get risk score
            risk_assessment = self.calculate_risk_score(ticker)
            risk_score = risk_assessment['risk_score']
            
            # Adjust max risk percentage based on risk score
            risk_adjustment = 1 - ((risk_score - 1) / 9)  # 1=full allocation, 10=minimal allocation
            adjusted_risk_pct = max_risk_pct * risk_adjustment
            
            # Calculate dollar amount
            dollar_amount = portfolio_value * (adjusted_risk_pct / 100)
            
            # Calculate number of shares
            current_price = self.tech_analyst.getData(ticker)[0]['Close'].iloc[-1]  # Get only the historical data
            shares = int(dollar_amount / current_price)
            
            return {
                'ticker': ticker,
                'risk_score': risk_score,
                'risk_category': risk_assessment['risk_category'],
                'recommended_position': dollar_amount,
                'shares': shares,
                'current_price': current_price
            }
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return None

# Example usage
if __name__ == "__main__":
    risk_mgr = risk_manager_agent()
    
    # Example risk assessment for Apple
    ticker = "AAPL"
    risk_assessment = risk_mgr.calculate_risk_score(ticker)
    print(f"Risk Score for {ticker}: {risk_assessment['risk_score']:.2f} ({risk_assessment['risk_category']})")
    
    # Example position sizing
    portfolio_value = 100000
    position_recommendation = risk_mgr.recommend_position_size(ticker, portfolio_value)
    if position_recommendation:
        print(f"Recommended position size: ${position_recommendation['recommended_position']:.2f} ({position_recommendation['shares']} shares)")
    
    # Print individual metrics with None handling
    beta = risk_mgr.calculate_beta(ticker)
    print(f"Beta: {beta:.2f}" if beta is not None else "Beta: Data not available")
    
    volatility = risk_mgr.calculate_volatility(ticker)
    print(f"Volatility (Annual): {volatility*100:.2f}%" if volatility is not None else "Volatility: Data not available")
    
    var = risk_mgr.calculate_var(ticker)
    print(f"Value at Risk (95%): {var:.2f}%" if var is not None else "VaR: Data not available")
    
    max_dd = risk_mgr.calculate_max_drawdown(ticker)
    print(f"Maximum Drawdown: {max_dd:.2f}%" if max_dd is not None else "Max Drawdown: Data not available")
    
    sharpe = risk_mgr.calculate_sharpe_ratio(ticker)
    print(f"Sharpe Ratio: {sharpe:.2f}" if sharpe is not None else "Sharpe Ratio: Data not available")
    
    sortino = risk_mgr.calculate_sortino_ratio(ticker)
    print(f"Sortino Ratio: {sortino:.2f}" if sortino is not None else "Sortino Ratio: Data not available")