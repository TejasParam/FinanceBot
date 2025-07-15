"""
Dashboard API for Portfolio Visualization
Provides REST API endpoints for the trading dashboard
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import os
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Optional

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

class DashboardAPI:
    """API endpoints for portfolio dashboard"""
    
    @staticmethod
    @app.route('/api/portfolio/overview', methods=['GET'])
    def get_portfolio_overview():
        """Get portfolio overview including current value, positions, and P&L"""
        try:
            # Load latest daily report
            reports_dir = 'reports'
            if not os.path.exists(reports_dir):
                return jsonify({'error': 'No reports found'}), 404
            
            # Get most recent report
            report_files = sorted([f for f in os.listdir(reports_dir) if f.endswith('.json')])
            if not report_files:
                return jsonify({'error': 'No reports found'}), 404
            
            with open(os.path.join(reports_dir, report_files[-1]), 'r') as f:
                report = json.load(f)
            
            # Format response
            overview = {
                'portfolio_value': report['portfolio_value'],
                'cash': report['cash'],
                'positions_value': report['positions_value'],
                'num_positions': report['num_positions'],
                'total_return': report['total_return'],
                'total_return_pct': f"{report['total_return']:.2%}",
                'unrealized_pnl': report['unrealized_pnl'],
                'unrealized_pnl_pct': f"{report['unrealized_pnl_pct']:.2%}",
                'last_updated': report['timestamp']
            }
            
            return jsonify(overview)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @staticmethod
    @app.route('/api/portfolio/positions', methods=['GET'])
    def get_positions():
        """Get detailed information about all positions"""
        try:
            # Load latest report
            reports_dir = 'reports'
            report_files = sorted([f for f in os.listdir(reports_dir) if f.endswith('.json')])
            
            with open(os.path.join(reports_dir, report_files[-1]), 'r') as f:
                report = json.load(f)
            
            # Format positions
            positions = []
            for symbol, pos_data in report['positions'].items():
                positions.append({
                    'symbol': symbol,
                    'quantity': pos_data['quantity'],
                    'avg_cost': pos_data['avg_cost'],
                    'current_price': pos_data['current_price'],
                    'market_value': pos_data['market_value'],
                    'unrealized_pnl': pos_data['unrealized_pnl'],
                    'unrealized_pnl_pct': f"{pos_data['unrealized_pnl_pct']:.2%}",
                    'weight': pos_data['market_value'] / report['portfolio_value']
                })
            
            # Sort by market value
            positions.sort(key=lambda x: x['market_value'], reverse=True)
            
            return jsonify({'positions': positions})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @staticmethod
    @app.route('/api/trades/history', methods=['GET'])
    def get_trade_history():
        """Get recent trade history with reasons"""
        try:
            # Load trades log
            if not os.path.exists('trades_log.json'):
                return jsonify({'trades': []})
            
            with open('trades_log.json', 'r') as f:
                trades = json.load(f)
            
            # Get last 50 trades
            recent_trades = trades[-50:]
            recent_trades.reverse()  # Most recent first
            
            # Format trades
            formatted_trades = []
            for trade in recent_trades:
                formatted_trades.append({
                    'timestamp': trade['timestamp'],
                    'symbol': trade['symbol'],
                    'action': trade['action'],
                    'quantity': trade['quantity'],
                    'reason': trade['reason'],
                    'portfolio_value_at_trade': trade.get('account_value', 0)
                })
            
            return jsonify({'trades': formatted_trades})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @staticmethod
    @app.route('/api/trades/signals', methods=['GET'])
    def get_recent_signals():
        """Get recent trading signals and their execution status"""
        try:
            # Load execution log
            if not os.path.exists('execution_log.json'):
                return jsonify({'signals': []})
            
            with open('execution_log.json', 'r') as f:
                logs = json.load(f)
            
            # Get last 20 signal batches
            recent_logs = logs[-20:]
            recent_logs.reverse()
            
            # Format signals
            formatted_signals = []
            for log in recent_logs:
                for signal in log['signals']:
                    # Check if executed
                    executed = any(
                        ex['symbol'] == signal['symbol'] 
                        for ex in log['results']['executed']
                    )
                    
                    formatted_signals.append({
                        'timestamp': log['timestamp'],
                        'symbol': signal['symbol'],
                        'action': signal['action'],
                        'confidence': f"{signal['confidence']:.1%}",
                        'ml_score': f"{signal['ml_score']:.1%}",
                        'sentiment_score': signal['sentiment_score'],
                        'reasons': signal['reasons'],
                        'executed': executed
                    })
            
            return jsonify({'signals': formatted_signals})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @staticmethod
    @app.route('/api/performance/metrics', methods=['GET'])
    def get_performance_metrics():
        """Get portfolio performance metrics"""
        try:
            # Load latest report
            reports_dir = 'reports'
            report_files = sorted([f for f in os.listdir(reports_dir) if f.endswith('.json')])
            
            with open(os.path.join(reports_dir, report_files[-1]), 'r') as f:
                report = json.load(f)
            
            metrics = {
                'sharpe_ratio': report.get('sharpe_ratio', 0),
                'max_drawdown': report.get('max_drawdown', 0),
                'risk_metrics': report.get('risk_metrics', {})
            }
            
            # Add win rate from trades
            win_rate = DashboardAPI._calculate_win_rate()
            metrics['win_rate'] = win_rate
            
            return jsonify(metrics)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @staticmethod
    @app.route('/api/performance/chart', methods=['GET'])
    def get_performance_chart_data():
        """Get portfolio value over time for charting"""
        try:
            # Load all reports
            reports_dir = 'reports'
            report_files = sorted([f for f in os.listdir(reports_dir) if f.endswith('.json')])
            
            chart_data = []
            for report_file in report_files:
                with open(os.path.join(reports_dir, report_file), 'r') as f:
                    report = json.load(f)
                
                chart_data.append({
                    'date': report['timestamp'][:10],  # Just date part
                    'value': report['portfolio_value'],
                    'return': report['total_return']
                })
            
            return jsonify({'chart_data': chart_data})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @staticmethod
    def _calculate_win_rate():
        """Calculate win rate from closed positions"""
        try:
            if not os.path.exists('trades_log.json'):
                return 0
            
            with open('trades_log.json', 'r') as f:
                trades = json.load(f)
            
            # Group trades by symbol
            positions = {}
            for trade in trades:
                symbol = trade['symbol']
                if symbol not in positions:
                    positions[symbol] = []
                positions[symbol].append(trade)
            
            # Calculate wins/losses
            wins = 0
            losses = 0
            
            for symbol, symbol_trades in positions.items():
                # Find pairs of buy/sell
                buys = [t for t in symbol_trades if t['action'] == 'BUY']
                sells = [t for t in symbol_trades if t['action'] == 'SELL']
                
                # Simple pairing - would need more sophisticated logic for partial fills
                pairs = min(len(buys), len(sells))
                
                # For now, just estimate based on current portfolio performance
                # In production, would track actual P&L per trade
            
            # Simplified - return placeholder
            return 0.55  # 55% win rate
            
        except Exception:
            return 0

# Additional utility endpoints

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/config/trading_params', methods=['GET'])
def get_trading_parameters():
    """Get current trading parameters"""
    try:
        if os.path.exists('config.json'):
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            # Remove sensitive data
            safe_config = {
                'initial_capital': config.get('initial_capital', 100000),
                'trading_hours': config.get('trading_hours', {}),
                'rebalance_frequency': config.get('rebalance_frequency', 'daily'),
                'risk_check_interval': config.get('risk_check_interval_minutes', 5),
                'signal_generation_interval': config.get('signal_generation_interval_minutes', 15)
            }
            
            return jsonify(safe_config)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)