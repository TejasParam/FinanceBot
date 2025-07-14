import ray
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable, Optional
import time
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import agents
from agents.technical_agent import TechnicalAnalysisAgent
from agents.fundamental_agent import FundamentalAnalysisAgent
from agents.sentiment_agent import SentimentAnalysisAgent
from agents.ml_agent_enhanced import EnhancedMLAgent
from agents.regime_agent import MarketRegimeAgent

class ParallelAgentExecutor:
    """
    Parallel execution framework for multiple agents using Ray or multiprocessing
    """
    
    def __init__(self, use_ray: bool = True, n_workers: Optional[int] = None):
        self.use_ray = use_ray and self._check_ray_available()
        self.n_workers = n_workers or multiprocessing.cpu_count()
        
        if self.use_ray:
            # Initialize Ray if not already initialized
            if not ray.is_initialized():
                ray.init(num_cpus=self.n_workers, ignore_reinit_error=True)
                logger.info(f"Ray initialized with {self.n_workers} workers")
        else:
            logger.info(f"Using multiprocessing with {self.n_workers} workers")
    
    def _check_ray_available(self) -> bool:
        """Check if Ray is available and can be initialized"""
        try:
            import ray
            return True
        except ImportError:
            logger.warning("Ray not available, falling back to multiprocessing")
            return False
    
    def execute_agents_parallel(self, 
                              ticker: str,
                              agents: List[Any],
                              timeout: int = 30) -> Dict[str, Any]:
        """
        Execute multiple agents in parallel
        
        Args:
            ticker: Stock ticker symbol
            agents: List of agent instances to execute
            timeout: Maximum time to wait for results (seconds)
        
        Returns:
            Dictionary with results from each agent
        """
        
        if self.use_ray:
            return self._execute_with_ray(ticker, agents, timeout)
        else:
            return self._execute_with_multiprocessing(ticker, agents, timeout)
    
    def _execute_with_ray(self, ticker: str, agents: List[Any], timeout: int) -> Dict[str, Any]:
        """Execute agents using Ray for distributed computing"""
        
        # Create Ray actors for each agent
        @ray.remote
        class AgentActor:
            def __init__(self, agent_class, agent_name):
                self.agent = agent_class()
                self.agent_name = agent_name
            
            def analyze(self, ticker):
                start_time = time.time()
                try:
                    result = self.agent.analyze(ticker)
                    execution_time = time.time() - start_time
                    return {
                        'agent': self.agent_name,
                        'result': result,
                        'execution_time': execution_time,
                        'status': 'success'
                    }
                except Exception as e:
                    return {
                        'agent': self.agent_name,
                        'error': str(e),
                        'execution_time': time.time() - start_time,
                        'status': 'error'
                    }
        
        # Create actors
        actors = []
        for agent in agents:
            agent_class = type(agent)
            agent_name = agent_class.__name__
            actor = AgentActor.remote(agent_class, agent_name)
            actors.append(actor)
        
        # Execute analyses in parallel
        futures = [actor.analyze.remote(ticker) for actor in actors]
        
        # Wait for results with timeout
        ready_futures, remaining_futures = ray.wait(
            futures,
            num_returns=len(futures),
            timeout=timeout
        )
        
        # Collect results
        results = {}
        for future in ready_futures:
            try:
                result = ray.get(future)
                results[result['agent']] = result
            except Exception as e:
                logger.error(f"Error getting result: {e}")
        
        # Handle timeouts
        for future in remaining_futures:
            ray.cancel(future)
            # Add timeout result
            results[f"Unknown_Agent_{len(results)}"] = {
                'status': 'timeout',
                'execution_time': timeout
            }
        
        return results
    
    def _execute_with_multiprocessing(self, ticker: str, agents: List[Any], timeout: int) -> Dict[str, Any]:
        """Execute agents using multiprocessing"""
        
        def run_agent(agent_info):
            """Worker function to run a single agent"""
            agent_class, agent_name, ticker = agent_info
            start_time = time.time()
            
            try:
                agent = agent_class()
                result = agent.analyze(ticker)
                execution_time = time.time() - start_time
                
                return {
                    'agent': agent_name,
                    'result': result,
                    'execution_time': execution_time,
                    'status': 'success'
                }
            except Exception as e:
                return {
                    'agent': agent_name,
                    'error': str(e),
                    'execution_time': time.time() - start_time,
                    'status': 'error'
                }
        
        # Prepare agent information
        agent_infos = [
            (type(agent), type(agent).__name__, ticker)
            for agent in agents
        ]
        
        results = {}
        
        # Use ProcessPoolExecutor for CPU-bound tasks
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            future_to_agent = {
                executor.submit(run_agent, info): info[1]
                for info in agent_infos
            }
            
            # Wait for completion with timeout
            for future in as_completed(future_to_agent, timeout=timeout):
                agent_name = future_to_agent[future]
                try:
                    result = future.result()
                    results[agent_name] = result
                except Exception as e:
                    results[agent_name] = {
                        'status': 'error',
                        'error': str(e)
                    }
        
        return results
    
    def batch_analyze_tickers(self,
                            tickers: List[str],
                            agent_classes: List[type],
                            batch_size: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Analyze multiple tickers in parallel batches
        
        Args:
            tickers: List of ticker symbols
            agent_classes: List of agent classes to use
            batch_size: Number of tickers to process in parallel
        
        Returns:
            Dictionary with results for each ticker
        """
        
        all_results = {}
        
        # Process tickers in batches
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            
            if self.use_ray:
                batch_results = self._batch_analyze_ray(batch_tickers, agent_classes)
            else:
                batch_results = self._batch_analyze_multiprocessing(batch_tickers, agent_classes)
            
            all_results.update(batch_results)
            
            logger.info(f"Completed batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size}")
        
        return all_results
    
    def _batch_analyze_ray(self, tickers: List[str], agent_classes: List[type]) -> Dict[str, Dict[str, Any]]:
        """Batch analysis using Ray"""
        
        @ray.remote
        def analyze_ticker(ticker, agent_classes):
            """Analyze a single ticker with all agents"""
            results = {}
            
            for agent_class in agent_classes:
                try:
                    agent = agent_class()
                    result = agent.analyze(ticker)
                    results[agent_class.__name__] = {
                        'result': result,
                        'status': 'success'
                    }
                except Exception as e:
                    results[agent_class.__name__] = {
                        'error': str(e),
                        'status': 'error'
                    }
            
            return ticker, results
        
        # Submit all ticker analyses
        futures = [analyze_ticker.remote(ticker, agent_classes) for ticker in tickers]
        
        # Collect results
        batch_results = {}
        for future in futures:
            try:
                ticker, results = ray.get(future)
                batch_results[ticker] = results
            except Exception as e:
                logger.error(f"Error in batch analysis: {e}")
        
        return batch_results
    
    def _batch_analyze_multiprocessing(self, tickers: List[str], agent_classes: List[type]) -> Dict[str, Dict[str, Any]]:
        """Batch analysis using multiprocessing"""
        
        def analyze_ticker_worker(args):
            """Worker function for analyzing a ticker"""
            ticker, agent_classes = args
            results = {}
            
            for agent_class in agent_classes:
                try:
                    agent = agent_class()
                    result = agent.analyze(ticker)
                    results[agent_class.__name__] = {
                        'result': result,
                        'status': 'success'
                    }
                except Exception as e:
                    results[agent_class.__name__] = {
                        'error': str(e),
                        'status': 'error'
                    }
            
            return ticker, results
        
        # Prepare arguments
        args_list = [(ticker, agent_classes) for ticker in tickers]
        
        batch_results = {}
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(analyze_ticker_worker, args) for args in args_list]
            
            for future in as_completed(futures):
                try:
                    ticker, results = future.result()
                    batch_results[ticker] = results
                except Exception as e:
                    logger.error(f"Error in batch analysis: {e}")
        
        return batch_results
    
    def optimize_portfolio_parallel(self,
                                  tickers: List[str],
                                  optimization_strategies: List[str],
                                  constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run multiple portfolio optimization strategies in parallel
        """
        
        from portfolio_optimizer import EnhancedPortfolioOptimizer
        
        @ray.remote
        def optimize_strategy(tickers, strategy, constraints):
            """Optimize portfolio with a specific strategy"""
            optimizer = EnhancedPortfolioOptimizer()
            
            start_time = time.time()
            result = optimizer.optimize_portfolio(
                tickers,
                strategy=strategy,
                constraints=constraints
            )
            execution_time = time.time() - start_time
            
            return {
                'strategy': strategy,
                'result': result,
                'execution_time': execution_time
            }
        
        if self.use_ray:
            # Ray parallel execution
            futures = [
                optimize_strategy.remote(tickers, strategy, constraints)
                for strategy in optimization_strategies
            ]
            
            results = {}
            for future in futures:
                try:
                    result = ray.get(future)
                    results[result['strategy']] = result
                except Exception as e:
                    logger.error(f"Error in portfolio optimization: {e}")
        else:
            # Multiprocessing execution
            results = {}
            
            def optimize_worker(strategy):
                optimizer = EnhancedPortfolioOptimizer()
                start_time = time.time()
                
                try:
                    result = optimizer.optimize_portfolio(
                        tickers,
                        strategy=strategy,
                        constraints=constraints
                    )
                    return {
                        'strategy': strategy,
                        'result': result,
                        'execution_time': time.time() - start_time
                    }
                except Exception as e:
                    return {
                        'strategy': strategy,
                        'error': str(e),
                        'execution_time': time.time() - start_time
                    }
            
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(optimize_worker, strategy): strategy
                    for strategy in optimization_strategies
                }
                
                for future in as_completed(futures):
                    strategy = futures[future]
                    try:
                        result = future.result()
                        results[strategy] = result
                    except Exception as e:
                        results[strategy] = {'error': str(e)}
        
        # Find best strategy
        best_strategy = None
        best_sharpe = -np.inf
        
        for strategy, data in results.items():
            if 'result' in data and 'sharpe_ratio' in data['result']:
                if data['result']['sharpe_ratio'] > best_sharpe:
                    best_sharpe = data['result']['sharpe_ratio']
                    best_strategy = strategy
        
        return {
            'results': results,
            'best_strategy': best_strategy,
            'best_sharpe_ratio': best_sharpe,
            'total_strategies_tested': len(results)
        }
    
    def shutdown(self):
        """Shutdown parallel execution framework"""
        if self.use_ray and ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown complete")


class AsyncAgentExecutor:
    """
    Asynchronous execution for I/O bound operations
    """
    
    def __init__(self):
        self.session = None
    
    async def fetch_data_async(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple tickers asynchronously"""
        import aiohttp
        import yfinance as yf
        
        async def fetch_ticker(session, ticker):
            """Fetch data for a single ticker"""
            try:
                # In practice, you'd use an async API here
                # For demonstration, we'll use yfinance in a thread pool
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    None,
                    lambda: yf.download(ticker, period="1y", progress=False)
                )
                return ticker, data
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
                return ticker, None
        
        # Create session and fetch all tickers
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_ticker(session, ticker) for ticker in tickers]
            results = await asyncio.gather(*tasks)
        
        # Convert to dictionary
        return {ticker: data for ticker, data in results if data is not None}
    
    async def analyze_news_async(self, tickers: List[str]) -> Dict[str, Any]:
        """Analyze news for multiple tickers asynchronously"""
        
        async def analyze_ticker_news(ticker):
            """Analyze news for a single ticker"""
            try:
                # Simulated async news analysis
                await asyncio.sleep(0.1)  # Simulate API call
                
                return {
                    'ticker': ticker,
                    'sentiment': np.random.choice(['bullish', 'bearish', 'neutral']),
                    'score': np.random.uniform(-1, 1),
                    'articles_analyzed': np.random.randint(5, 20)
                }
            except Exception as e:
                return {
                    'ticker': ticker,
                    'error': str(e)
                }
        
        # Run all analyses concurrently
        tasks = [analyze_ticker_news(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks)
        
        return {result['ticker']: result for result in results}


# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = 0  # Would use psutil in practice
        
        result = func(*args, **kwargs)
        
        execution_time = time.time() - start_time
        
        logger.info(f"{func.__name__} completed in {execution_time:.2f} seconds")
        
        return result
    
    return wrapper


# Example usage
if __name__ == "__main__":
    # Initialize parallel executor
    executor = ParallelAgentExecutor(use_ray=True, n_workers=4)
    
    # Example 1: Parallel agent execution
    print("Example 1: Parallel Agent Execution")
    agents = [
        TechnicalAnalysisAgent(),
        FundamentalAnalysisAgent(),
        SentimentAnalysisAgent(),
        EnhancedMLAgent(),
        MarketRegimeAgent()
    ]
    
    results = executor.execute_agents_parallel("AAPL", agents, timeout=30)
    
    for agent_name, result in results.items():
        if result['status'] == 'success':
            print(f"{agent_name}: Completed in {result['execution_time']:.2f}s")
        else:
            print(f"{agent_name}: {result['status']}")
    
    # Example 2: Batch ticker analysis
    print("\nExample 2: Batch Ticker Analysis")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    agent_classes = [TechnicalAnalysisAgent, SentimentAnalysisAgent]
    
    batch_results = executor.batch_analyze_tickers(tickers, agent_classes, batch_size=3)
    
    for ticker, agents_results in batch_results.items():
        print(f"\n{ticker}:")
        for agent, result in agents_results.items():
            print(f"  {agent}: {result['status']}")
    
    # Example 3: Parallel portfolio optimization
    print("\nExample 3: Parallel Portfolio Optimization")
    portfolio_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    strategies = ['mean_variance', 'max_sharpe', 'min_volatility', 'risk_parity']
    
    optimization_results = executor.optimize_portfolio_parallel(
        portfolio_tickers,
        strategies
    )
    
    print(f"Best strategy: {optimization_results['best_strategy']}")
    print(f"Best Sharpe ratio: {optimization_results['best_sharpe_ratio']:.2f}")
    
    # Example 4: Async operations
    print("\nExample 4: Async Operations")
    async_executor = AsyncAgentExecutor()
    
    async def run_async_example():
        # Fetch data asynchronously
        data = await async_executor.fetch_data_async(["AAPL", "MSFT", "GOOGL"])
        print(f"Fetched data for {len(data)} tickers")
        
        # Analyze news asynchronously
        news_results = await async_executor.analyze_news_async(["AAPL", "MSFT", "GOOGL"])
        for ticker, analysis in news_results.items():
            if 'sentiment' in analysis:
                print(f"{ticker}: {analysis['sentiment']} (score: {analysis['score']:.2f})")
    
    # Run async example
    asyncio.run(run_async_example())
    
    # Cleanup
    executor.shutdown()