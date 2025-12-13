# src/backtest.py - Backtest trading strategy and evaluate performance

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score
import os
from datetime import datetime
from config import INITIAL_CAPITAL, CONFIDENCE_THRESHOLD, VERBOSE

class MetricsCalculator:
    """Calculate performance metrics"""
    
    @staticmethod
    def calculate_regression_metrics(y_true, y_pred):
        """Calculate regression metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.abs(y_true))) * 100
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MAPE': float(mape),
            'R2': float(r2)
        }
    
    @staticmethod
    def calculate_classification_metrics(y_true, y_pred, y_pred_proba=None):
        """Calculate classification metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'Accuracy': float(accuracy),
            'Precision': float(precision),
            'Recall': float(recall),
            'F1': float(f1)
        }
        
        if y_pred_proba is not None:
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_true, y_pred_proba)
                metrics['AUC'] = float(auc)
            except:
                metrics['AUC'] = None
        
        return metrics
    
    @staticmethod
    def calculate_directional_accuracy(actual_prices, predicted_prices):
        """Calculate directional accuracy"""
        actual_direction = np.diff(actual_prices) > 0
        pred_direction = np.diff(predicted_prices) > 0
        
        accuracy = np.mean(actual_direction == pred_direction) * 100
        return float(accuracy)

class BacktestEngine:
    """Backtest trading strategy"""
    
    def __init__(self, initial_capital=INITIAL_CAPITAL, 
                 confidence_threshold=CONFIDENCE_THRESHOLD, verbose=VERBOSE):
        self.initial_capital = initial_capital
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        
        self.trades = []
        self.portfolio_values = [initial_capital]
        self.capital = initial_capital
    
    def backtest_strategy(self, test_data, predictions):
        """Backtest trading strategy on test data"""
        position = None
        trades = []
        portfolio_values = [self.initial_capital]
        
        capital = self.initial_capital
        trade_num = 0
        
        for i in range(len(test_data) - 1):
            current_price = test_data['Close'].iloc[i]
            next_price = test_data['Close'].iloc[i + 1]
            
            # Get signal and predicted price
            if isinstance(predictions, dict) and 'signal' in predictions:
                signal = predictions['signal']
                predicted_price = predictions.get('predicted_price', current_price)
            else:
                signal = 'HOLD'
                predicted_price = current_price
            
            # Execute trade
            if signal == 'BUY' and position is None:
                position = {
                    'entry_price': current_price,
                    'entry_date': test_data['Date'].iloc[i],
                    'predicted_target': predicted_price
                }
                trade_num += 1
                trades.append({
                    'trade_num': trade_num,
                    'action': 'BUY',
                    'price': current_price,
                    'date': test_data['Date'].iloc[i],
                    'reason': 'Entry signal'
                })
            
            elif signal == 'SELL' and position is not None:
                exit_price = next_price
                profit = exit_price - position['entry_price']
                profit_pct = (profit / position['entry_price']) * 100
                
                capital += profit
                
                trades.append({
                    'trade_num': trade_num,
                    'action': 'SELL',
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'date': test_data['Date'].iloc[i],
                    'reason': 'Exit signal'
                })
                
                position = None
            
            # Update portfolio value
            if position is not None:
                unrealized = (next_price - position['entry_price']) * 1
                portfolio_values.append(capital + unrealized)
            else:
                portfolio_values.append(capital)
        
        # Close any open position at end
        if position is not None:
            final_price = test_data['Close'].iloc[-1]
            profit = final_price - position['entry_price']
            capital += profit
            trades.append({
                'trade_num': trade_num,
                'action': 'SELL',
                'entry_price': position['entry_price'],
                'exit_price': final_price,
                'profit': profit,
                'profit_pct': (profit / position['entry_price']) * 100,
                'date': test_data['Date'].iloc[-1],
                'reason': 'End of backtest'
            })
        
        return {
            'trades': trades,
            'portfolio_values': portfolio_values,
            'final_capital': capital
        }
    
    def calculate_trading_metrics(self, backtest_results):
        """Calculate trading performance metrics"""
        trades = backtest_results['trades']
        portfolio_values = np.array(backtest_results['portfolio_values'])
        
        # Filter only completed trades
        completed_trades = [t for t in trades if t['action'] == 'SELL']
        
        total_return = (backtest_results['final_capital'] - self.initial_capital) / self.initial_capital
        
        if len(completed_trades) > 0:
            wins = sum(1 for t in completed_trades if t['profit_pct'] > 0)
            losses = sum(1 for t in completed_trades if t['profit_pct'] < 0)
            win_rate = wins / len(completed_trades) if len(completed_trades) > 0 else 0
            
            avg_win = np.mean([t['profit_pct'] for t in completed_trades if t['profit_pct'] > 0]) if wins > 0 else 0
            avg_loss = np.mean([t['profit_pct'] for t in completed_trades if t['profit_pct'] < 0]) if losses > 0 else 0
            
            profit_factor = abs(sum(t['profit'] for t in completed_trades if t['profit'] > 0) / 
                               sum(t['profit'] for t in completed_trades if t['profit'] < 0)) if losses > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Volatility metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = (np.mean(returns) * 252 - 0.02) / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
        
        # Max drawdown
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar = (np.mean(returns) * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': float(total_return),
            'annual_return': float(np.mean(returns) * 252),
            'num_trades': len(completed_trades),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar)
        }

class EvaluationReporter:
    """Generate evaluation reports"""

    def __init__(self, verbose=True):
        self.verbose = verbose

    def generate_model_evaluation_report(
        self,
        y_test,
        y_pred_price,
        y_pred_direction,
        y_pred_direction_proba
    ):
        """Generate comprehensive model evaluation report"""

        regression_metrics = MetricsCalculator.calculate_regression_metrics(
            y_test, y_pred_price
        )

        y_test_direction = (y_test > 0).astype(int)
        classification_metrics = MetricsCalculator.calculate_classification_metrics(
            y_test_direction,
            y_pred_direction,
            y_pred_direction_proba
        )

        directional_acc = MetricsCalculator.calculate_directional_accuracy(
            y_test, y_pred_price
        )

        return {
            "regression_metrics": regression_metrics,
            "classification_metrics": classification_metrics,
            "directional_accuracy": directional_acc,
            "timestamp": datetime.now().isoformat()
        }

    def print_evaluation_report(self, report):
        """Print formatted evaluation report"""

        print("\n" + "=" * 70)
        print("MODEL EVALUATION REPORT")
        print("=" * 70)

        print("\n📊 REGRESSION METRICS")
        print("-" * 70)
        for k, v in report["regression_metrics"].items():
            print(f"  {k:20s}: {v:.4f}")

        print("\n🎯 CLASSIFICATION METRICS")
        print("-" * 70)
        for k, v in report["classification_metrics"].items():
            print(f"  {k:20s}: {v:.4f}")

        print("\n📈 DIRECTIONAL ACCURACY")
        print("-" * 70)
        print(f"  {report['directional_accuracy']:.2f}%")

        print("\n⏰ Generated:", report["timestamp"])
        print("=" * 70)

    def print_backtest_report(self, trading_metrics, trades=None):
        """Print formatted backtesting report"""

        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)

        print("\n💰 RETURNS")
        print("-" * 70)
        print(f"  Total Return:        {trading_metrics['total_return']:+.2%}")
        print(f"  Annual Return:       {trading_metrics['annual_return']:+.2%}")

        print("\n📊 TRADING STATS")
        print("-" * 70)
        print(f"  Trades:             {trading_metrics['num_trades']}")
        print(f"  Win Rate:           {trading_metrics['win_rate']:.2%}")
        print(f"  Avg Win:            {trading_metrics['avg_win']:+.2f}%")
        print(f"  Avg Loss:           {trading_metrics['avg_loss']:+.2f}%")
        print(f"  Profit Factor:      {trading_metrics['profit_factor']:.2f}")

        print("\n⚠️ RISK METRICS")
        print("-" * 70)
        print(f"  Volatility:         {trading_metrics['volatility']:.4f}")
        print(f"  Sharpe Ratio:       {trading_metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:       {trading_metrics['max_drawdown']:.2%}")
        print(f"  Calmar Ratio:       {trading_metrics['calmar_ratio']:.2f}")

        if trades:
            print("\n📋 RECENT TRADES")
            print("-" * 70)
            for t in trades[-5:]:
                if t["action"] == "SELL":
                    print(
                        f"  {t['date']} SELL @ ${t['exit_price']:.2f} "
                        f"({t['profit_pct']:+.2f}%)"
                    )

        print("=" * 70)

    def save_evaluation_report(self, report, ticker, output_dir="results"):
        """Save evaluation & backtest report to file"""

        import os

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{ticker}_evaluation_report.txt")

        with open(filepath, "w") as f:
            f.write("=" * 70 + "\n")
            f.write(f"MODEL EVALUATION REPORT - {ticker}\n")
            f.write("=" * 70 + "\n\n")

            for section, metrics in report.items():
                f.write(section.upper().replace("_", " ") + "\n")
                f.write("-" * 70 + "\n")

                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        if isinstance(v, float):
                            f.write(f"{k:25s}: {v:.4f}\n")
                        else:
                            f.write(f"{k:25s}: {v}\n")
                else:
                    f.write(str(metrics) + "\n")

                f.write("\n")

            f.write("=" * 70 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 70 + "\n")

        if self.verbose:
            print(f"✓ Report saved to {filepath}")

        return filepath
