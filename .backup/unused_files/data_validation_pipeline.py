import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityLevel(Enum):
    """Data quality levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INVALID = "invalid"

@dataclass
class ValidationRule:
    """Data validation rule definition"""
    name: str
    check_function: Callable
    severity: str  # 'error', 'warning', 'info'
    description: str
    auto_fix: Optional[Callable] = None

@dataclass
class ValidationResult:
    """Result of a validation check"""
    rule_name: str
    passed: bool
    severity: str
    message: str
    affected_data: Optional[Any] = None
    fix_applied: bool = False

class DataValidationPipeline:
    """
    Comprehensive data validation and quality assurance pipeline
    """
    
    def __init__(self):
        self.validation_rules = self._initialize_rules()
        self.validation_history = []
        self.quality_metrics = {}
        
    def _initialize_rules(self) -> List[ValidationRule]:
        """Initialize default validation rules"""
        return [
            # Completeness checks
            ValidationRule(
                name="no_missing_prices",
                check_function=self._check_missing_prices,
                severity="error",
                description="Check for missing price data",
                auto_fix=self._fix_missing_prices
            ),
            ValidationRule(
                name="no_missing_volume",
                check_function=self._check_missing_volume,
                severity="warning",
                description="Check for missing volume data",
                auto_fix=self._fix_missing_volume
            ),
            
            # Validity checks
            ValidationRule(
                name="positive_prices",
                check_function=self._check_positive_prices,
                severity="error",
                description="Ensure all prices are positive"
            ),
            ValidationRule(
                name="reasonable_price_changes",
                check_function=self._check_price_changes,
                severity="warning",
                description="Check for unreasonable price movements",
                auto_fix=self._fix_price_spikes
            ),
            
            # Consistency checks
            ValidationRule(
                name="ohlc_consistency",
                check_function=self._check_ohlc_consistency,
                severity="error",
                description="Ensure OHLC relationships are valid"
            ),
            ValidationRule(
                name="chronological_order",
                check_function=self._check_chronological_order,
                severity="error",
                description="Ensure data is in chronological order"
            ),
            
            # Statistical checks
            ValidationRule(
                name="outlier_detection",
                check_function=self._check_outliers,
                severity="warning",
                description="Detect statistical outliers",
                auto_fix=self._fix_outliers
            ),
            ValidationRule(
                name="data_staleness",
                check_function=self._check_staleness,
                severity="info",
                description="Check if data is up-to-date"
            ),
            
            # Market hours checks
            ValidationRule(
                name="trading_hours",
                check_function=self._check_trading_hours,
                severity="info",
                description="Verify data is from trading hours"
            ),
            
            # Cross-validation checks
            ValidationRule(
                name="volume_price_correlation",
                check_function=self._check_volume_price_correlation,
                severity="info",
                description="Check volume-price relationship"
            )
        ]
    
    def validate_data(self, 
                     data: pd.DataFrame,
                     ticker: str,
                     auto_fix: bool = True,
                     custom_rules: Optional[List[ValidationRule]] = None) -> Dict[str, Any]:
        """
        Validate financial data with comprehensive checks
        
        Args:
            data: DataFrame with financial data
            ticker: Stock ticker symbol
            auto_fix: Whether to apply automatic fixes
            custom_rules: Additional validation rules
        
        Returns:
            Validation report with results and quality metrics
        """
        
        logger.info(f"Starting data validation for {ticker}")
        
        # Combine default and custom rules
        rules = self.validation_rules + (custom_rules or [])
        
        # Run validation checks
        results = []
        data_copy = data.copy()
        
        for rule in rules:
            try:
                result = rule.check_function(data_copy, ticker)
                
                # Apply auto-fix if available and enabled
                if not result.passed and auto_fix and rule.auto_fix:
                    logger.info(f"Applying auto-fix for {rule.name}")
                    data_copy = rule.auto_fix(data_copy, result)
                    
                    # Re-run check after fix
                    result_after_fix = rule.check_function(data_copy, ticker)
                    result_after_fix.fix_applied = True
                    results.append(result_after_fix)
                else:
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Error in validation rule {rule.name}: {e}")
                results.append(ValidationResult(
                    rule_name=rule.name,
                    passed=False,
                    severity="error",
                    message=f"Validation error: {str(e)}"
                ))
        
        # Calculate quality metrics
        quality_score = self._calculate_quality_score(results)
        quality_level = self._determine_quality_level(quality_score)
        
        # Create validation report
        report = {
            'ticker': ticker,
            'timestamp': datetime.now(),
            'total_rules': len(rules),
            'passed_rules': sum(1 for r in results if r.passed),
            'failed_rules': sum(1 for r in results if not r.passed),
            'quality_score': quality_score,
            'quality_level': quality_level.value,
            'results': results,
            'fixes_applied': sum(1 for r in results if r.fix_applied),
            'cleaned_data': data_copy if auto_fix else None,
            'summary': self._generate_summary(results)
        }
        
        # Store in history
        self.validation_history.append({
            'ticker': ticker,
            'timestamp': datetime.now(),
            'quality_score': quality_score,
            'quality_level': quality_level.value
        })
        
        return report
    
    def _check_missing_prices(self, data: pd.DataFrame, ticker: str) -> ValidationResult:
        """Check for missing price data"""
        price_columns = ['Open', 'High', 'Low', 'Close']
        existing_columns = [col for col in price_columns if col in data.columns]
        
        if not existing_columns:
            return ValidationResult(
                rule_name="no_missing_prices",
                passed=False,
                severity="error",
                message="No price columns found in data"
            )
        
        missing_count = data[existing_columns].isna().sum().sum()
        
        if missing_count > 0:
            return ValidationResult(
                rule_name="no_missing_prices",
                passed=False,
                severity="error",
                message=f"Found {missing_count} missing price values",
                affected_data=data[data[existing_columns].isna().any(axis=1)]
            )
        
        return ValidationResult(
            rule_name="no_missing_prices",
            passed=True,
            severity="error",
            message="No missing price data found"
        )
    
    def _fix_missing_prices(self, data: pd.DataFrame, result: ValidationResult) -> pd.DataFrame:
        """Fix missing prices using forward fill then backward fill"""
        price_columns = ['Open', 'High', 'Low', 'Close']
        existing_columns = [col for col in price_columns if col in data.columns]
        
        # Forward fill then backward fill
        data[existing_columns] = data[existing_columns].fillna(method='ffill').fillna(method='bfill')
        
        # If still missing, use interpolation
        data[existing_columns] = data[existing_columns].interpolate(method='linear')
        
        return data
    
    def _check_missing_volume(self, data: pd.DataFrame, ticker: str) -> ValidationResult:
        """Check for missing volume data"""
        if 'Volume' not in data.columns:
            return ValidationResult(
                rule_name="no_missing_volume",
                passed=True,
                severity="warning",
                message="Volume column not found"
            )
        
        missing_count = data['Volume'].isna().sum()
        zero_count = (data['Volume'] == 0).sum()
        
        if missing_count > 0 or zero_count > len(data) * 0.1:  # More than 10% zeros
            return ValidationResult(
                rule_name="no_missing_volume",
                passed=False,
                severity="warning",
                message=f"Found {missing_count} missing and {zero_count} zero volume values"
            )
        
        return ValidationResult(
            rule_name="no_missing_volume",
            passed=True,
            severity="warning",
            message="Volume data is complete"
        )
    
    def _fix_missing_volume(self, data: pd.DataFrame, result: ValidationResult) -> pd.DataFrame:
        """Fix missing volume using average volume"""
        if 'Volume' in data.columns:
            # Calculate average volume excluding zeros and NaN
            avg_volume = data['Volume'][data['Volume'] > 0].mean()
            
            # Fill missing values
            data['Volume'] = data['Volume'].fillna(avg_volume)
            
            # Replace zeros with average volume
            data.loc[data['Volume'] == 0, 'Volume'] = avg_volume
        
        return data
    
    def _check_positive_prices(self, data: pd.DataFrame, ticker: str) -> ValidationResult:
        """Check that all prices are positive"""
        price_columns = ['Open', 'High', 'Low', 'Close']
        existing_columns = [col for col in price_columns if col in data.columns]
        
        negative_prices = (data[existing_columns] <= 0).any().any()
        
        if negative_prices:
            negative_rows = data[(data[existing_columns] <= 0).any(axis=1)]
            return ValidationResult(
                rule_name="positive_prices",
                passed=False,
                severity="error",
                message=f"Found {len(negative_rows)} rows with non-positive prices",
                affected_data=negative_rows
            )
        
        return ValidationResult(
            rule_name="positive_prices",
            passed=True,
            severity="error",
            message="All prices are positive"
        )
    
    def _check_price_changes(self, data: pd.DataFrame, ticker: str) -> ValidationResult:
        """Check for unreasonable price changes (>20% in a day)"""
        if 'Close' not in data.columns:
            return ValidationResult(
                rule_name="reasonable_price_changes",
                passed=True,
                severity="warning",
                message="Close price column not found"
            )
        
        price_changes = data['Close'].pct_change()
        large_changes = price_changes.abs() > 0.20  # 20% threshold
        
        if large_changes.any():
            affected_dates = data.index[large_changes].tolist()
            return ValidationResult(
                rule_name="reasonable_price_changes",
                passed=False,
                severity="warning",
                message=f"Found {large_changes.sum()} days with >20% price changes",
                affected_data=affected_dates
            )
        
        return ValidationResult(
            rule_name="reasonable_price_changes",
            passed=True,
            severity="warning",
            message="All price changes are within reasonable bounds"
        )
    
    def _fix_price_spikes(self, data: pd.DataFrame, result: ValidationResult) -> pd.DataFrame:
        """Fix price spikes using rolling median"""
        if 'Close' in data.columns:
            # Calculate rolling median
            rolling_median = data['Close'].rolling(window=5, center=True).median()
            
            # Identify spikes
            price_changes = data['Close'].pct_change().abs()
            spikes = price_changes > 0.20
            
            # Replace spikes with rolling median
            data.loc[spikes, 'Close'] = rolling_median[spikes]
            
            # Adjust OHLC if needed
            for col in ['Open', 'High', 'Low']:
                if col in data.columns:
                    data.loc[spikes, col] = data.loc[spikes, 'Close']
        
        return data
    
    def _check_ohlc_consistency(self, data: pd.DataFrame, ticker: str) -> ValidationResult:
        """Check OHLC relationships (High >= Low, High >= Open/Close, etc.)"""
        required_columns = ['Open', 'High', 'Low', 'Close']
        existing_columns = [col for col in required_columns if col in data.columns]
        
        if len(existing_columns) < 4:
            return ValidationResult(
                rule_name="ohlc_consistency",
                passed=True,
                severity="error",
                message="Not all OHLC columns present"
            )
        
        # Check relationships
        inconsistencies = []
        
        # High should be >= all other prices
        high_violations = (data['High'] < data[['Open', 'Low', 'Close']].max(axis=1))
        if high_violations.any():
            inconsistencies.append(f"High < max(Open,Low,Close): {high_violations.sum()} rows")
        
        # Low should be <= all other prices
        low_violations = (data['Low'] > data[['Open', 'High', 'Close']].min(axis=1))
        if low_violations.any():
            inconsistencies.append(f"Low > min(Open,High,Close): {low_violations.sum()} rows")
        
        if inconsistencies:
            return ValidationResult(
                rule_name="ohlc_consistency",
                passed=False,
                severity="error",
                message=f"OHLC inconsistencies found: {'; '.join(inconsistencies)}"
            )
        
        return ValidationResult(
            rule_name="ohlc_consistency",
            passed=True,
            severity="error",
            message="OHLC relationships are consistent"
        )
    
    def _check_chronological_order(self, data: pd.DataFrame, ticker: str) -> ValidationResult:
        """Check if data is in chronological order"""
        if not isinstance(data.index, pd.DatetimeIndex):
            return ValidationResult(
                rule_name="chronological_order",
                passed=False,
                severity="error",
                message="Index is not a DatetimeIndex"
            )
        
        # Check if sorted
        is_sorted = data.index.equals(data.index.sort_values())
        
        if not is_sorted:
            return ValidationResult(
                rule_name="chronological_order",
                passed=False,
                severity="error",
                message="Data is not in chronological order"
            )
        
        # Check for duplicates
        duplicates = data.index.duplicated().sum()
        if duplicates > 0:
            return ValidationResult(
                rule_name="chronological_order",
                passed=False,
                severity="error",
                message=f"Found {duplicates} duplicate timestamps"
            )
        
        return ValidationResult(
            rule_name="chronological_order",
            passed=True,
            severity="error",
            message="Data is in proper chronological order"
        )
    
    def _check_outliers(self, data: pd.DataFrame, ticker: str) -> ValidationResult:
        """Detect statistical outliers using IQR method"""
        if 'Close' not in data.columns:
            return ValidationResult(
                rule_name="outlier_detection",
                passed=True,
                severity="warning",
                message="Close price column not found"
            )
        
        # Calculate returns
        returns = data['Close'].pct_change().dropna()
        
        # IQR method
        Q1 = returns.quantile(0.25)
        Q3 = returns.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = (returns < lower_bound) | (returns > upper_bound)
        
        if outliers.any():
            outlier_dates = returns[outliers].index.tolist()
            return ValidationResult(
                rule_name="outlier_detection",
                passed=False,
                severity="warning",
                message=f"Found {outliers.sum()} statistical outliers",
                affected_data=outlier_dates
            )
        
        return ValidationResult(
            rule_name="outlier_detection",
            passed=True,
            severity="warning",
            message="No statistical outliers detected"
        )
    
    def _fix_outliers(self, data: pd.DataFrame, result: ValidationResult) -> pd.DataFrame:
        """Fix outliers using winsorization"""
        if 'Close' in data.columns:
            # Calculate returns
            returns = data['Close'].pct_change()
            
            # Winsorize at 1st and 99th percentiles
            lower_limit = returns.quantile(0.01)
            upper_limit = returns.quantile(0.99)
            
            # Clip returns
            returns_clipped = returns.clip(lower=lower_limit, upper=upper_limit)
            
            # Reconstruct prices
            data['Close'] = data['Close'].iloc[0] * (1 + returns_clipped).cumprod()
            
            # Adjust other price columns proportionally
            for col in ['Open', 'High', 'Low']:
                if col in data.columns:
                    ratio = data[col] / data['Close']
                    data[col] = data['Close'] * ratio
        
        return data
    
    def _check_staleness(self, data: pd.DataFrame, ticker: str) -> ValidationResult:
        """Check if data is up-to-date"""
        if data.empty:
            return ValidationResult(
                rule_name="data_staleness",
                passed=False,
                severity="info",
                message="No data available"
            )
        
        last_date = data.index[-1]
        current_date = pd.Timestamp.now()
        
        # Account for weekends and market holidays
        business_days_old = pd.bdate_range(last_date, current_date).size - 1
        
        if business_days_old > 5:  # More than 5 business days old
            return ValidationResult(
                rule_name="data_staleness",
                passed=False,
                severity="info",
                message=f"Data is {business_days_old} business days old"
            )
        
        return ValidationResult(
            rule_name="data_staleness",
            passed=True,
            severity="info",
            message=f"Data is current (last update: {last_date.date()})"
        )
    
    def _check_trading_hours(self, data: pd.DataFrame, ticker: str) -> ValidationResult:
        """Verify data is from regular trading hours"""
        if not isinstance(data.index, pd.DatetimeIndex):
            return ValidationResult(
                rule_name="trading_hours",
                passed=True,
                severity="info",
                message="Cannot check trading hours without datetime index"
            )
        
        # Check if data has time component
        if not any(data.index.time != pd.Timestamp('00:00:00').time()):
            return ValidationResult(
                rule_name="trading_hours",
                passed=True,
                severity="info",
                message="Data appears to be daily (no intraday timestamps)"
            )
        
        # Check for after-hours data (simplified check)
        after_hours = 0
        for timestamp in data.index:
            hour = timestamp.hour
            if hour < 9 or hour >= 16:  # Before 9:30 AM or after 4:00 PM
                after_hours += 1
        
        if after_hours > 0:
            return ValidationResult(
                rule_name="trading_hours",
                passed=False,
                severity="info",
                message=f"Found {after_hours} data points outside regular trading hours"
            )
        
        return ValidationResult(
            rule_name="trading_hours",
            passed=True,
            severity="info",
            message="All data from regular trading hours"
        )
    
    def _check_volume_price_correlation(self, data: pd.DataFrame, ticker: str) -> ValidationResult:
        """Check if volume and price movements are correlated (they often are)"""
        if 'Close' not in data.columns or 'Volume' not in data.columns:
            return ValidationResult(
                rule_name="volume_price_correlation",
                passed=True,
                severity="info",
                message="Cannot check correlation without price and volume data"
            )
        
        # Calculate absolute price changes and volume changes
        price_changes = data['Close'].pct_change().abs()
        volume_changes = data['Volume'].pct_change().abs()
        
        # Remove infinities and NaN
        mask = np.isfinite(price_changes) & np.isfinite(volume_changes)
        
        if mask.sum() < 10:
            return ValidationResult(
                rule_name="volume_price_correlation",
                passed=True,
                severity="info",
                message="Insufficient data for correlation check"
            )
        
        # Calculate correlation
        correlation = price_changes[mask].corr(volume_changes[mask])
        
        # Typically expect positive correlation
        if correlation < -0.3:
            return ValidationResult(
                rule_name="volume_price_correlation",
                passed=False,
                severity="info",
                message=f"Unusual negative correlation between price and volume changes: {correlation:.2f}"
            )
        
        return ValidationResult(
            rule_name="volume_price_correlation",
            passed=True,
            severity="info",
            message=f"Normal volume-price correlation: {correlation:.2f}"
        )
    
    def _calculate_quality_score(self, results: List[ValidationResult]) -> float:
        """Calculate overall data quality score"""
        if not results:
            return 0.0
        
        # Weight by severity
        severity_weights = {
            'error': 1.0,
            'warning': 0.5,
            'info': 0.2
        }
        
        total_weight = 0
        weighted_score = 0
        
        for result in results:
            weight = severity_weights.get(result.severity, 0.5)
            total_weight += weight
            if result.passed:
                weighted_score += weight
        
        return (weighted_score / total_weight * 100) if total_weight > 0 else 0
    
    def _determine_quality_level(self, score: float) -> DataQualityLevel:
        """Determine quality level from score"""
        if score >= 95:
            return DataQualityLevel.HIGH
        elif score >= 80:
            return DataQualityLevel.MEDIUM
        elif score >= 60:
            return DataQualityLevel.LOW
        else:
            return DataQualityLevel.INVALID
    
    def _generate_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate summary of validation results"""
        summary = {
            'errors': [],
            'warnings': [],
            'info': [],
            'fixes_applied': []
        }
        
        for result in results:
            if not result.passed:
                if result.severity == 'error':
                    summary['errors'].append(result.message)
                elif result.severity == 'warning':
                    summary['warnings'].append(result.message)
                else:
                    summary['info'].append(result.message)
            
            if result.fix_applied:
                summary['fixes_applied'].append(result.rule_name)
        
        return summary
    
    def generate_quality_report(self, ticker: str) -> Dict[str, Any]:
        """Generate quality report for a ticker based on history"""
        ticker_history = [h for h in self.validation_history if h['ticker'] == ticker]
        
        if not ticker_history:
            return {
                'ticker': ticker,
                'message': 'No validation history available'
            }
        
        scores = [h['quality_score'] for h in ticker_history]
        
        return {
            'ticker': ticker,
            'validations_performed': len(ticker_history),
            'average_quality_score': np.mean(scores),
            'min_quality_score': np.min(scores),
            'max_quality_score': np.max(scores),
            'quality_trend': 'improving' if len(scores) > 1 and scores[-1] > scores[0] else 'stable',
            'last_validation': ticker_history[-1]['timestamp'],
            'current_quality_level': ticker_history[-1]['quality_level']
        }
    
    def export_validation_rules(self, filepath: str):
        """Export validation rules to JSON"""
        rules_data = []
        
        for rule in self.validation_rules:
            rules_data.append({
                'name': rule.name,
                'severity': rule.severity,
                'description': rule.description,
                'has_auto_fix': rule.auto_fix is not None
            })
        
        with open(filepath, 'w') as f:
            json.dump(rules_data, f, indent=2)
        
        logger.info(f"Exported {len(rules_data)} validation rules to {filepath}")


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = DataValidationPipeline()
    
    # Create sample data with issues
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    data = pd.DataFrame({
        'Open': np.random.randn(len(dates)).cumsum() + 100,
        'High': np.random.randn(len(dates)).cumsum() + 101,
        'Low': np.random.randn(len(dates)).cumsum() + 99,
        'Close': np.random.randn(len(dates)).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # Introduce some data quality issues
    data.iloc[50:52, data.columns.get_loc('Close')] = np.nan  # Missing values
    data.iloc[100, data.columns.get_loc('Close')] = -10  # Negative price
    data.iloc[150, data.columns.get_loc('Volume')] = 0  # Zero volume
    data.iloc[200, data.columns.get_loc('High')] = data.iloc[200, data.columns.get_loc('Low')] - 1  # OHLC inconsistency
    
    # Run validation
    report = pipeline.validate_data(data, 'TEST', auto_fix=True)
    
    # Print results
    print(f"Data Quality Score: {report['quality_score']:.1f}%")
    print(f"Quality Level: {report['quality_level']}")
    print(f"Passed Rules: {report['passed_rules']}/{report['total_rules']}")
    print(f"Fixes Applied: {report['fixes_applied']}")
    
    print("\nSummary:")
    for key, messages in report['summary'].items():
        if messages:
            print(f"\n{key.upper()}:")
            for msg in messages:
                print(f"  - {msg}")
    
    # Generate quality report
    quality_report = pipeline.generate_quality_report('TEST')
    print(f"\nQuality Report: {quality_report}")
    
    # Export rules
    pipeline.export_validation_rules('validation_rules.json')