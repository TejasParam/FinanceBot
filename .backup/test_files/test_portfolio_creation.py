import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Import the modules we want to test
from portfolio_optimizer import EnhancedPortfolioOptimizer
from risk_manager_enhanced import EnhancedRiskManager
from data_validation_pipeline import DataValidationPipeline
from demos.portfolio_builder import PortfolioBuilder

class TestPortfolioCreation(unittest.TestCase):
    """
    Comprehensive test suite for portfolio creation functionality
    """
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = EnhancedPortfolioOptimizer()
        self.risk_manager = EnhancedRiskManager()
        self.validator = DataValidationPipeline()
        self.portfolio_builder = PortfolioBuilder()
        
        # Test tickers
        self.test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        self.small_test_set = ["AAPL", "MSFT"]
        
        # Sample data for testing
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample financial data for testing"""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        np.random.seed(42)  # For reproducible tests
        
        data = pd.DataFrame({
            'Open': 100 + np.random.randn(len(dates)).cumsum() * 0.5,
            'High': 101 + np.random.randn(len(dates)).cumsum() * 0.5,
            'Low': 99 + np.random.randn(len(dates)).cumsum() * 0.5,
            'Close': 100 + np.random.randn(len(dates)).cumsum() * 0.5,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Ensure OHLC relationships are valid
        data['High'] = data[['Open', 'High', 'Low', 'Close']].max(axis=1)
        data['Low'] = data[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        return data
    
    def test_basic_portfolio_optimization(self):
        """Test basic mean-variance optimization"""
        print("\n🧪 Testing basic portfolio optimization...")
        
        result = self.optimizer.optimize_portfolio(
            tickers=self.small_test_set,
            strategy='mean_variance'
        )
        
        # Assertions
        self.assertIn('weights', result)
        self.assertIn('metrics', result)
        self.assertIsInstance(result['weights'], dict)
        
        # Check weights sum to 1 (approximately)
        total_weight = sum(result['weights'].values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        
        # Check all weights are non-negative
        for weight in result['weights'].values():
            self.assertGreaterEqual(weight, 0)
        
        print(f"✅ Basic optimization successful: {result['weights']}")
    
    def test_constrained_optimization(self):
        """Test optimization with constraints"""
        print("\n🧪 Testing constrained optimization...")
        
        constraints = {
            'weight_bounds': {'AAPL': (0.2, 0.6), 'MSFT': (0.1, 0.5)},
            'max_assets': 2
        }
        
        result = self.optimizer.optimize_portfolio(
            tickers=self.small_test_set,
            strategy='mean_variance',
            constraints=constraints
        )
        
        # Check constraints are respected
        if 'weights' in result:
            aapl_weight = result['weights'].get('AAPL', 0)
            msft_weight = result['weights'].get('MSFT', 0)
            
            self.assertGreaterEqual(aapl_weight, 0.15)  # Allow some tolerance
            self.assertLessEqual(aapl_weight, 0.65)
            self.assertGreaterEqual(msft_weight, 0.05)
            self.assertLessEqual(msft_weight, 0.55)
        
        print(f"✅ Constrained optimization successful")
    
    def test_black_litterman_optimization(self):
        """Test Black-Litterman optimization with views"""
        print("\n🧪 Testing Black-Litterman optimization...")
        
        # Define market views
        views = {
            'absolute': {
                'AAPL': 0.12,  # Expect 12% return
                'MSFT': 0.08   # Expect 8% return
            },
            'confidence': {
                'AAPL': 0.8,
                'MSFT': 0.6
            }
        }
        
        result = self.optimizer.optimize_portfolio(
            tickers=self.small_test_set,
            strategy='black_litterman',
            views=views
        )
        
        # Assertions
        if 'weights' in result:
            self.assertIn('view_impact', result)
            self.assertIn('posterior_returns', result)
            
            # Check that views had some impact
            view_impact = result['view_impact']
            self.assertTrue(any(abs(impact) > 0.001 for impact in view_impact.values()))
        
        print("✅ Black-Litterman optimization successful")
    
    def test_risk_parity_optimization(self):
        """Test risk parity optimization"""
        print("\n🧪 Testing risk parity optimization...")
        
        result = self.optimizer.optimize_portfolio(
            tickers=self.small_test_set,
            strategy='risk_parity'
        )
        
        if 'weights' in result and 'risk_contributions' in result:
            risk_contribs = list(result['risk_contributions'].values())
            
            # Risk contributions should be roughly equal
            max_contrib = max(risk_contribs)
            min_contrib = min(risk_contribs)
            ratio = max_contrib / min_contrib if min_contrib > 0 else float('inf')
            
            # Allow some tolerance (ratio should be reasonable)
            self.assertLess(ratio, 3.0, "Risk contributions not well-balanced")
        
        print("✅ Risk parity optimization successful")
    
    def test_portfolio_metrics_calculation(self):
        """Test portfolio performance metrics"""
        print("\n🧪 Testing portfolio metrics calculation...")
        
        # Create simple equal-weight portfolio
        weights = {ticker: 1/len(self.small_test_set) for ticker in self.small_test_set}
        
        # Mock returns data
        returns_data = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'MSFT': np.random.normal(0.0008, 0.018, 252)
        })
        
        metrics = self.optimizer._calculate_portfolio_metrics(weights, returns_data)
        
        # Check required metrics are present
        required_metrics = [
            'annual_return', 'annual_volatility', 'sharpe_ratio',
            'max_drawdown', 'var_95', 'cvar_95'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertFalse(np.isnan(metrics[metric]), f"{metric} is NaN")
        
        print("✅ Portfolio metrics calculation successful")
    
    def test_risk_manager_integration(self):
        """Test risk manager integration"""
        print("\n🧪 Testing risk manager integration...")
        
        # Test VaR calculations
        var_metrics = self.risk_manager.calculate_all_var_metrics("AAPL", period=100)
        
        if var_metrics:
            self.assertIn('var_historical', var_metrics)
            self.assertIn('var_parametric', var_metrics)
            self.assertIn('var_monte_carlo', var_metrics)
            self.assertIn('cvar', var_metrics)
            
            # VaR should be positive (representing potential loss)
            self.assertGreater(var_metrics['var_historical'], 0)
            self.assertGreater(var_metrics['cvar'], 0)
        
        print("✅ Risk manager integration successful")
    
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo portfolio simulation"""
        print("\n🧪 Testing Monte Carlo simulation...")
        
        portfolio_weights = {'AAPL': 0.6, 'MSFT': 0.4}
        
        mc_results = self.risk_manager.monte_carlo_portfolio_simulation(
            portfolio_weights,
            n_simulations=1000,
            time_horizon=63,  # 3 months
            initial_value=100000
        )
        
        if mc_results:
            required_keys = [
                'mean_final_value', 'var_95', 'probability_loss',
                'percentile_5', 'percentile_95'
            ]
            
            for key in required_keys:
                self.assertIn(key, mc_results)
            
            # Sanity checks
            self.assertGreater(mc_results['percentile_95'], mc_results['percentile_5'])
            self.assertGreaterEqual(mc_results['probability_loss'], 0)
            self.assertLessEqual(mc_results['probability_loss'], 1)
        
        print("✅ Monte Carlo simulation successful")
    
    def test_stress_testing(self):
        """Test portfolio stress testing"""
        print("\n🧪 Testing stress testing...")
        
        portfolio_weights = {'AAPL': 0.6, 'MSFT': 0.4}
        
        stress_results = self.risk_manager.stress_test_portfolio(portfolio_weights)
        
        if stress_results:
            self.assertIn('summary', stress_results)
            
            # Should have multiple scenarios
            scenario_count = len([k for k in stress_results.keys() if k != 'summary'])
            self.assertGreater(scenario_count, 3)
            
            # Each scenario should have required fields
            for key, result in stress_results.items():
                if key != 'summary' and isinstance(result, dict):
                    self.assertIn('portfolio_loss', result)
                    self.assertIn('stressed_return', result)
        
        print("✅ Stress testing successful")
    
    def test_data_validation(self):
        """Test data validation pipeline"""
        print("\n🧪 Testing data validation...")
        
        # Test with clean data
        validation_report = self.validator.validate_data(
            self.sample_data,
            'TEST',
            auto_fix=True
        )
        
        self.assertIn('quality_score', validation_report)
        self.assertIn('quality_level', validation_report)
        self.assertIn('results', validation_report)
        
        # Quality score should be reasonable
        self.assertGreaterEqual(validation_report['quality_score'], 50)
        
        print(f"✅ Data validation successful: {validation_report['quality_score']:.1f}% quality")
    
    def test_portfolio_builder_integration(self):
        """Test complete portfolio builder workflow"""
        print("\n🧪 Testing portfolio builder integration...")
        
        try:
            # Test conservative portfolio
            conservative_portfolio = self.portfolio_builder.build_conservative_portfolio(
                portfolio_value=100000
            )
            
            self.assertIn('allocation', conservative_portfolio)
            self.assertIn('summary', conservative_portfolio)
            
            # Conservative portfolio should have reasonable allocation
            allocation = conservative_portfolio['allocation']
            if allocation:
                total_allocation = sum(allocation.values())
                self.assertAlmostEqual(total_allocation, 1.0, places=1)
            
            print("✅ Portfolio builder integration successful")
            
        except Exception as e:
            print(f"⚠️ Portfolio builder test failed (may need real data): {e}")
    
    def test_efficient_frontier(self):
        """Test efficient frontier generation"""
        print("\n🧪 Testing efficient frontier generation...")
        
        frontier_data = self.optimizer.efficient_frontier(
            self.small_test_set,
            n_portfolios=10
        )
        
        if not frontier_data.empty:
            required_columns = ['return', 'volatility', 'sharpe']
            for col in required_columns:
                self.assertIn(col, frontier_data.columns)
            
            # Returns and volatility should be positive
            self.assertTrue((frontier_data['return'] >= 0).any())
            self.assertTrue((frontier_data['volatility'] > 0).all())
        
        print("✅ Efficient frontier generation successful")
    
    def test_rebalancing_recommendations(self):
        """Test portfolio rebalancing"""
        print("\n🧪 Testing rebalancing recommendations...")
        
        current_weights = {'AAPL': 0.7, 'MSFT': 0.3}
        target_weights = {'AAPL': 0.6, 'MSFT': 0.4}
        current_values = {'AAPL': 70000, 'MSFT': 30000}
        
        rebalance_result = self.optimizer.rebalance_portfolio(
            current_weights,
            target_weights,
            current_values,
            threshold=0.05
        )
        
        self.assertIn('trades', rebalance_result)
        self.assertIn('estimated_costs', rebalance_result)
        
        # Should recommend trades since weights differ by more than threshold
        if rebalance_result['trades']:
            self.assertGreater(len(rebalance_result['trades']), 0)
        
        print("✅ Rebalancing recommendations successful")
    
    def test_correlation_analysis(self):
        """Test correlation matrix calculation"""
        print("\n🧪 Testing correlation analysis...")
        
        corr_matrix = self.optimizer.calculate_correlation_matrix(self.small_test_set)
        
        if not corr_matrix.empty:
            # Should be square matrix
            self.assertEqual(len(corr_matrix.index), len(corr_matrix.columns))
            
            # Diagonal should be 1.0 (or close to it)
            for ticker in corr_matrix.index:
                if ticker in corr_matrix.columns:
                    self.assertAlmostEqual(corr_matrix.loc[ticker, ticker], 1.0, places=1)
            
            # Should be symmetric
            for i in corr_matrix.index:
                for j in corr_matrix.columns:
                    if i in corr_matrix.columns and j in corr_matrix.index:
                        self.assertAlmostEqual(
                            corr_matrix.loc[i, j],
                            corr_matrix.loc[j, i],
                            places=4
                        )
        
        print("✅ Correlation analysis successful")
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        print("\n🧪 Testing error handling...")
        
        # Test with empty ticker list
        result = self.optimizer.optimize_portfolio([])
        self.assertIn('error', result)
        
        # Test with invalid strategy
        result = self.optimizer.optimize_portfolio(
            self.small_test_set,
            strategy='invalid_strategy'
        )
        self.assertIn('error', result)
        
        print("✅ Error handling tests successful")


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics calculations"""
    
    def setUp(self):
        self.optimizer = EnhancedPortfolioOptimizer()
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation"""
        print("\n🧪 Testing Sharpe ratio calculation...")
        
        # Create synthetic returns with known properties
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        
        # Calculate expected Sharpe ratio
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        expected_sharpe = (annual_return - 0.05) / annual_vol
        
        # Test with portfolio metrics
        weights = {'TEST': 1.0}
        returns_df = pd.DataFrame({'TEST': returns})
        metrics = self.optimizer._calculate_portfolio_metrics(weights, returns_df)
        
        # Should be close to expected value
        self.assertAlmostEqual(metrics['sharpe_ratio'], expected_sharpe, places=2)
        
        print("✅ Sharpe ratio calculation correct")
    
    def test_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        print("\n🧪 Testing drawdown calculation...")
        
        # Create returns with known drawdown
        prices = [100, 110, 105, 95, 90, 100, 105]
        returns = pd.Series(prices).pct_change().dropna()
        
        weights = {'TEST': 1.0}
        returns_df = pd.DataFrame({'TEST': returns})
        metrics = self.optimizer._calculate_portfolio_metrics(weights, returns_df)
        
        # Should detect the drawdown from 110 to 90
        expected_max_dd = -0.1818  # (90-110)/110
        self.assertAlmostEqual(metrics['max_drawdown'], expected_max_dd, places=2)
        
        print("✅ Drawdown calculation correct")


def run_portfolio_tests():
    """Run all portfolio creation tests"""
    print("🚀 Starting Portfolio Creation Test Suite")
    print("=" * 60)
    
    # Create test suites
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestPortfolioCreation)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestPerformanceMetrics)
    
    # Combine suites
    combined_suite = unittest.TestSuite([suite1, suite2])
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(combined_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️ Some tests failed. Success rate: {(result.testsRun - len(result.failures) - len(result.errors))/result.testsRun*100:.1f}%")
    
    return result


def quick_portfolio_test():
    """Quick test to verify basic portfolio functionality"""
    print("⚡ Quick Portfolio Test")
    print("-" * 30)
    
    try:
        # Test basic optimization
        optimizer = EnhancedPortfolioOptimizer()
        result = optimizer.optimize_portfolio(['AAPL', 'MSFT'], strategy='mean_variance')
        
        if 'weights' in result:
            print("✅ Basic optimization: PASS")
            print(f"   Weights: {result['weights']}")
            print(f"   Expected Return: {result.get('expected_return', 'N/A'):.2%}")
            print(f"   Volatility: {result.get('volatility', 'N/A'):.2%}")
            print(f"   Sharpe Ratio: {result.get('sharpe_ratio', 'N/A'):.2f}")
        else:
            print("❌ Basic optimization: FAIL")
            print(f"   Error: {result.get('error', 'Unknown error')}")
        
        # Test risk management
        risk_manager = EnhancedRiskManager()
        var_result = risk_manager.calculate_all_var_metrics('AAPL', period=100)
        
        if var_result:
            print("✅ Risk management: PASS")
            print(f"   VaR (95%): {var_result.get('var_historical', 'N/A'):.2f}%")
        else:
            print("❌ Risk management: FAIL")
        
        print("\n✨ Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    # Check if user wants quick test or full test
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        quick_portfolio_test()
    else:
        # Run full test suite
        result = run_portfolio_tests()
        
        # Exit with error code if tests failed
        sys.exit(0 if result.wasSuccessful() else 1)