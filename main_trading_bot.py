#!/usr/bin/env python3
"""
Main Trading Bot
Runs continuously to manage the Alpaca paper trading portfolio
"""

import os
import sys
import time
import logging
import json
import schedule
from datetime import datetime, time as dt_time
import argparse
from typing import Dict
from dotenv import load_dotenv

from alpaca_integration import AlpacaTradingBot
from automated_portfolio_manager import AutomatedPortfolioManager

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TradingBotRunner:
    """Main class to run the trading bot"""
    
    def __init__(self, config_file: str = 'config.json'):
        """Initialize the trading bot runner"""
        self.config = self._load_config(config_file)
        
        # Initialize Alpaca bot
        self.alpaca_bot = AlpacaTradingBot(
            api_key=self.config['alpaca_api_key'],
            secret_key=self.config['alpaca_secret_key'],
            paper=True  # Always use paper trading for safety
        )
        
        # Initialize portfolio manager
        self.portfolio_manager = AutomatedPortfolioManager(
            alpaca_bot=self.alpaca_bot,
            initial_capital=self.config.get('initial_capital', 100000)
        )
        
        # Trading schedule
        self.trading_enabled = True
        self.last_health_check = datetime.now()
        
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file or environment variables"""
        config = {}
        
        # Try to load from config file if it exists
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        # Use environment variables as fallback or override
        config['alpaca_api_key'] = os.getenv('alpaca_key', config.get('alpaca_api_key'))
        config['alpaca_secret_key'] = os.getenv('alpaca_secret', config.get('alpaca_secret_key'))
        
        # Check if we have the required credentials
        if not config.get('alpaca_api_key') or not config.get('alpaca_secret_key'):
            logger.error("Alpaca API credentials not found!")
            logger.info("Please set alpaca_key and alpaca_secret in .env file or create config.json")
            if not os.path.exists(config_file):
                self._create_template_config(config_file)
            sys.exit(1)
        
        # Set defaults for other config values
        config.setdefault('initial_capital', 100000)
        config.setdefault('trading_hours', {'start': '09:30', 'end': '16:00'})
        config.setdefault('rebalance_frequency', 'daily')
        config.setdefault('risk_check_interval_minutes', 5)
        config.setdefault('signal_generation_interval_minutes', 15)
        
        logger.info("Configuration loaded successfully")
        return config
    
    def _create_template_config(self, config_file: str):
        """Create a template configuration file"""
        template = {
            "alpaca_api_key": "YOUR_ALPACA_API_KEY",
            "alpaca_secret_key": "YOUR_ALPACA_SECRET_KEY",
            "initial_capital": 100000,
            "trading_hours": {
                "start": "09:30",
                "end": "16:00"
            },
            "rebalance_frequency": "daily",
            "risk_check_interval_minutes": 5,
            "signal_generation_interval_minutes": 15
        }
        
        with open(config_file, 'w') as f:
            json.dump(template, f, indent=2)
        
        logger.info(f"Template config created at {config_file}")
        logger.info("Please update with your Alpaca API credentials")
    
    def is_market_open(self) -> bool:
        """Check if market is open for trading"""
        now = datetime.now()
        current_time = now.time()
        
        # Get trading hours from config
        start_time = dt_time.fromisoformat(self.config['trading_hours']['start'])
        end_time = dt_time.fromisoformat(self.config['trading_hours']['end'])
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if within trading hours
        return start_time <= current_time <= end_time
    
    def health_check(self):
        """Perform system health check"""
        try:
            # Check account status
            account_info = self.alpaca_bot.get_account_info()
            
            if account_info['trading_blocked']:
                logger.error("Trading is blocked on account!")
                self.trading_enabled = False
                return False
            
            if account_info['account_blocked']:
                logger.error("Account is blocked!")
                self.trading_enabled = False
                return False
            
            # Log account status
            logger.info(f"Account Status - Value: ${account_info['portfolio_value']:,.2f}, "
                       f"Cash: ${account_info['cash']:,.2f}, "
                       f"Buying Power: ${account_info['buying_power']:,.2f}")
            
            self.last_health_check = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def generate_and_execute_signals(self):
        """Generate trading signals and execute them"""
        if not self.trading_enabled:
            logger.warning("Trading is disabled")
            return
        
        if not self.is_market_open():
            logger.info("Market is closed")
            return
        
        try:
            logger.info("Generating trading signals...")
            
            # Generate signals
            signals = self.portfolio_manager.generate_trading_signals()
            logger.info(f"Generated {len(signals)} trading signals")
            
            # Log signals
            for signal in signals:
                logger.info(f"Signal: {signal.symbol} - {signal.action} "
                           f"(Confidence: {signal.confidence:.2%}, "
                           f"ML: {signal.ml_score:.2%}, "
                           f"Sentiment: {signal.sentiment_score:.2f})")
            
            # Execute signals
            if signals:
                logger.info("Executing trading signals...")
                results = self.portfolio_manager.execute_signals(signals)
                
                logger.info(f"Executed: {len(results['executed'])} trades")
                logger.info(f"Failed: {len(results['failed'])} trades")
                logger.info(f"Skipped: {len(results['skipped'])} trades")
                
                # Save execution results
                self._save_execution_results(results, signals)
                
        except Exception as e:
            logger.error(f"Error in signal generation/execution: {e}")
    
    def manage_risk(self):
        """Run risk management checks"""
        if not self.trading_enabled:
            return
        
        try:
            logger.info("Running risk management checks...")
            
            # Check stop-losses and take-profits
            results = self.portfolio_manager.manage_risk()
            
            if results['stopped_out']:
                logger.info(f"Stopped out positions: {results['stopped_out']}")
            
            if results['took_profits']:
                logger.info(f"Took profits on: {results['took_profits']}")
                
        except Exception as e:
            logger.error(f"Error in risk management: {e}")
    
    def rebalance_portfolio(self):
        """Rebalance the portfolio"""
        if not self.trading_enabled:
            return
        
        if not self.is_market_open():
            return
        
        try:
            logger.info("Checking portfolio balance...")
            
            results = self.portfolio_manager.rebalance_portfolio()
            
            if results['rebalanced']:
                logger.info(f"Rebalanced positions: {results['rebalanced']}")
            else:
                logger.info("No rebalancing needed")
                
        except Exception as e:
            logger.error(f"Error in portfolio rebalancing: {e}")
    
    def generate_daily_report(self):
        """Generate and save daily portfolio report"""
        try:
            logger.info("Generating daily portfolio report...")
            
            report = self.portfolio_manager.generate_portfolio_report()
            
            # Save report
            report_file = f"reports/daily_report_{datetime.now().strftime('%Y%m%d')}.json"
            os.makedirs('reports', exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Daily report saved to {report_file}")
            
            # Log summary
            logger.info(f"Portfolio Value: ${report['portfolio_value']:,.2f}")
            logger.info(f"Total Return: {report['total_return']:.2%}")
            logger.info(f"Positions: {report['num_positions']}")
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
    
    def _save_execution_results(self, results: Dict, signals: list):
        """Save execution results for dashboard"""
        execution_log = {
            'timestamp': datetime.now().isoformat(),
            'signals': [
                {
                    'symbol': s.symbol,
                    'action': s.action,
                    'confidence': s.confidence,
                    'ml_score': s.ml_score,
                    'sentiment_score': s.sentiment_score,
                    'reasons': s.reasons
                }
                for s in signals
            ],
            'results': results
        }
        
        # Append to execution log
        log_file = 'execution_log.json'
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(execution_log)
            
            # Keep only last 100 entries
            logs = logs[-100:]
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving execution results: {e}")
    
    def setup_schedule(self):
        """Setup the trading schedule"""
        # Health check every 5 minutes
        schedule.every(5).minutes.do(self.health_check)
        
        # Risk management checks
        risk_interval = self.config.get('risk_check_interval_minutes', 5)
        schedule.every(risk_interval).minutes.do(self.manage_risk)
        
        # Signal generation
        signal_interval = self.config.get('signal_generation_interval_minutes', 15)
        schedule.every(signal_interval).minutes.do(self.generate_and_execute_signals)
        
        # Daily tasks
        schedule.every().day.at("09:15").do(self.health_check)
        schedule.every().day.at("15:45").do(self.rebalance_portfolio)
        schedule.every().day.at("16:30").do(self.generate_daily_report)
        
        logger.info("Trading schedule configured")
    
    def run(self):
        """Main run loop"""
        logger.info("Starting Alpaca Trading Bot...")
        
        # Initial health check
        if not self.health_check():
            logger.error("Initial health check failed!")
            return
        
        # Setup schedule
        self.setup_schedule()
        
        # Generate initial signals
        self.generate_and_execute_signals()
        
        logger.info("Trading bot is running. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Shutting down trading bot...")
            
            # Cancel any open orders
            self.alpaca_bot.cancel_all_orders()
            
            # Generate final report
            self.generate_daily_report()
            
            logger.info("Trading bot stopped")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Alpaca Trading Bot')
    parser.add_argument('--config', default='config.json', 
                       help='Path to configuration file')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (single iteration)')
    
    args = parser.parse_args()
    
    # Create bot runner
    bot = TradingBotRunner(config_file=args.config)
    
    if args.test:
        # Test mode - run once
        logger.info("Running in test mode...")
        bot.health_check()
        bot.generate_and_execute_signals()
        bot.generate_daily_report()
    else:
        # Normal mode - run continuously
        bot.run()

if __name__ == "__main__":
    main()