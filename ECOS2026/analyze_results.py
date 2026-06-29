#!/usr/bin/env python3
"""
ECOS2026 - Result Analysis and Comparison Script

This script demonstrates how to analyze and compare results from the ECOS2026 system.
Run this after jobs complete to generate comparison tables and visualizations.

Usage:
    python3 analyze_results.py --results-dir results --dataset dataset_name
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False


class ResultsAnalyzer:
    """Analyze ECOS2026 results from organized directory structure."""
    
    def __init__(self, results_dir: str, dataset_name: str):
        self.results_dir = Path(results_dir)
        self.dataset_name = dataset_name
        self.dataset_path = self.results_dir / dataset_name
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        print(f"Analyzing results from: {self.dataset_path}")
    
    def load_all_results(self) -> pd.DataFrame:
        """
        Load all results into a single DataFrame for easy analysis.
        
        Returns:
            DataFrame with columns: dataset, cpd_algo, min_segment, delta, 
                                   window, trend, forecast_algo, 
                                   rmse, mae, mape, cpd_tp, cpd_fp, cpd_fn
        """
        all_results = []
        
        # Iterate through all CPD configurations
        for cpd_config_dir in self.dataset_path.glob("cpd_*"):
            if not cpd_config_dir.is_dir():
                continue
            
            # Parse CPD config name: cpd_Pelt_ms28_d1_w28_t1
            config_parts = cpd_config_dir.name.split('_')
            cpd_algo = config_parts[1]
            
            # Parse other parameters from name
            params = {}
            for part in config_parts[2:]:
                if part.startswith('ms'):
                    params['min_segment'] = int(part[2:])
                elif part.startswith('d'):
                    params['delta'] = int(part[1:])
                elif part.startswith('w'):
                    params['window'] = int(part[1:])
                elif part.startswith('t'):
                    params['trend'] = int(part[1:])
            
            # Iterate through all forecast algorithms
            for forecast_dir in cpd_config_dir.glob("forecast_*"):
                if not forecast_dir.is_dir():
                    continue
                
                forecast_algo = forecast_dir.name.replace("forecast_", "")
                
                try:
                    # Load configuration
                    config_file = forecast_dir / "config.json"
                    if config_file.exists():
                        with open(config_file) as f:
                            config = json.load(f)
                    else:
                        config = {}
                    
                    # Load forecasting metrics
                    metrics_file = forecast_dir / "02_forecasting_metrics.json"
                    if metrics_file.exists():
                        with open(metrics_file) as f:
                            metrics = json.load(f)
                        forecast_metrics = metrics.get(forecast_algo, {})
                    else:
                        forecast_metrics = {}
                    
                    # Load CPD on original
                    cpd_original_file = forecast_dir / "01_cpd_original.json"
                    if cpd_original_file.exists():
                        with open(cpd_original_file) as f:
                            cpd_orig = json.load(f)
                        cpd_metrics = cpd_orig.get('original_cpd', {}).get(cpd_algo, {})
                    else:
                        cpd_metrics = {}
                    
                    # Compile result
                    result = {
                        'dataset': self.dataset_name,
                        'cpd_algo': cpd_algo,
                        'min_segment': params.get('min_segment', 0),
                        'delta': params.get('delta', 0),
                        'window': params.get('window', 0),
                        'trend': params.get('trend', 0),
                        'forecast_algo': forecast_algo,
                        'rmse': forecast_metrics.get('rmse', float('inf')),
                        'mae': forecast_metrics.get('mae', float('inf')),
                        'mape': forecast_metrics.get('mape', float('inf')),
                        'cpd_tp': cpd_metrics.get('tp', 0),
                        'cpd_fp': cpd_metrics.get('fp', 0),
                        'cpd_fn': cpd_metrics.get('fn', 0),
                        'cpd_precision': cpd_metrics.get('precision', 0),
                        'cpd_recall': cpd_metrics.get('recall', 0),
                        'cpd_f1': cpd_metrics.get('f1_score', 0),
                    }
                    all_results.append(result)
                
                except Exception as e:
                    print(f"  Warning: Failed to parse {forecast_dir}: {e}")
                    continue
        
        df = pd.DataFrame(all_results)
        print(f"\nLoaded {len(df)} result combinations")
        return df
    
    def compare_forecast_algorithms(self, df: pd.DataFrame, 
                                   cpd_algo: str, delta: int, trend: int) -> pd.DataFrame:
        """
        Compare all forecast algorithms for a specific CPD configuration.
        
        Args:
            df: Results dataframe
            cpd_algo: CPD algorithm name (e.g., "Pelt")
            delta: Delta value (e.g., 1)
            trend: Trend order (e.g., 1)
        
        Returns:
            Sorted DataFrame with comparison
        """
        subset = df[
            (df['cpd_algo'] == cpd_algo) &
            (df['delta'] == delta) &
            (df['trend'] == trend)
        ].copy()
        
        subset = subset.sort_values('rmse')
        print(f"\n{'='*80}")
        print(f"Forecast Algorithm Comparison")
        print(f"CPD: {cpd_algo}, Delta: {delta}, Trend: {trend}")
        print(f"{'='*80}")
        print(subset[['forecast_algo', 'rmse', 'mae', 'mape']].to_string(index=False))
        
        return subset
    
    def compare_trend_orders(self, df: pd.DataFrame,
                            cpd_algo: str, delta: int, 
                            forecast_algo: str) -> pd.DataFrame:
        """
        Compare different trend orders for a specific configuration.
        
        Args:
            df: Results dataframe
            cpd_algo: CPD algorithm
            delta: Delta value
            forecast_algo: Forecast algorithm
        
        Returns:
            Sorted DataFrame with comparison
        """
        subset = df[
            (df['cpd_algo'] == cpd_algo) &
            (df['delta'] == delta) &
            (df['forecast_algo'] == forecast_algo)
        ].copy()
        
        subset = subset.sort_values('rmse')
        print(f"\n{'='*80}")
        print(f"Trend Order Comparison")
        print(f"CPD: {cpd_algo}, Forecast: {forecast_algo}, Delta: {delta}")
        print(f"{'='*80}")
        print(subset[['trend', 'rmse', 'mae', 'mape']].to_string(index=False))
        
        return subset
    
    def compare_cpd_algorithms(self, df: pd.DataFrame,
                              delta: int, trend: int,
                              forecast_algo: str) -> pd.DataFrame:
        """
        Compare different CPD algorithms.
        
        Args:
            df: Results dataframe
            delta: Delta value
            trend: Trend order
            forecast_algo: Forecast algorithm
        
        Returns:
            Sorted DataFrame with comparison
        """
        subset = df[
            (df['delta'] == delta) &
            (df['trend'] == trend) &
            (df['forecast_algo'] == forecast_algo)
        ].copy()
        
        subset = subset.sort_values('rmse')
        print(f"\n{'='*80}")
        print(f"CPD Algorithm Comparison")
        print(f"Forecast: {forecast_algo}, Trend: {trend}, Delta: {delta}")
        print(f"{'='*80}")
        print(subset[['cpd_algo', 'rmse', 'mae', 'mape', 'cpd_f1']].to_string(index=False))
        
        return subset
    
    def best_configurations(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Find best configurations by RMSE."""
        best = df.nsmallest(top_n, 'rmse')
        print(f"\n{'='*80}")
        print(f"Top {top_n} Best Configurations (by RMSE)")
        print(f"{'='*80}")
        print(best[['cpd_algo', 'trend', 'forecast_algo', 'rmse', 
                    'mae', 'mape', 'cpd_f1']].to_string(index=False))
        return best
    
    def algorithm_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summary statistics per forecast algorithm."""
        summary = df.groupby('forecast_algo')[['rmse', 'mae', 'mape']].agg([
            'mean', 'std', 'min', 'max'
        ]).round(4)
        
        print(f"\n{'='*80}")
        print(f"Forecast Algorithm Summary Statistics")
        print(f"{'='*80}")
        print(summary)
        
        return summary
    
    def cpd_algorithm_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summary statistics per CPD algorithm."""
        summary = df.groupby('cpd_algo')[['rmse', 'cpd_f1']].agg([
            'mean', 'std', 'min', 'max'
        ]).round(4)
        
        print(f"\n{'='*80}")
        print(f"CPD Algorithm Summary Statistics")
        print(f"{'='*80}")
        print(summary)
        
        return summary
    
    def plot_algorithm_comparison(self, df: pd.DataFrame, metric: str = 'rmse'):
        """Create boxplot comparing forecast algorithms."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available - skipping plots")
            return
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='forecast_algo', y=metric)
        plt.title(f'Forecast Algorithm Comparison ({metric.upper()})')
        plt.xlabel('Forecast Algorithm')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'forecast_algo_comparison_{metric}.png', dpi=150)
        print(f"Saved: forecast_algo_comparison_{metric}.png")
        plt.close()
    
    def plot_trend_comparison(self, df: pd.DataFrame, metric: str = 'rmse'):
        """Create boxplot comparing trend orders."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available - skipping plots")
            return
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='trend', y=metric)
        plt.title(f'Trend Order Comparison ({metric.upper()})')
        plt.xlabel('Trend Order')
        plt.ylabel(metric.upper())
        plt.tight_layout()
        plt.savefig(f'trend_comparison_{metric}.png', dpi=150)
        print(f"Saved: trend_comparison_{metric}.png")
        plt.close()


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Analyze ECOS2026 Results')
    parser.add_argument('--results-dir', type=str, default=str(script_dir / 'results' / 'raw'),
                       help='Results directory (default: ECOS2026/results/raw)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name to analyze')
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(args.results_dir, args.dataset)
    
    # Load all results
    df = analyzer.load_all_results()
    
    if len(df) == 0:
        print("No results found!")
        return
    
    # Run analyses
    print("\n" + "="*80)
    print("ECOS2026 RESULTS ANALYSIS")
    print("="*80)
    
    # Summary statistics
    analyzer.algorithm_summary(df)
    analyzer.cpd_algorithm_summary(df)
    
    # Find best
    analyzer.best_configurations(df, top_n=15)
    
    # Detailed comparisons (examples)
    if len(df) > 0:
        # Get first available CPD algo
        first_cpd = df['cpd_algo'].iloc[0]
        first_delta = df['delta'].iloc[0]
        first_trend = df['trend'].iloc[0]
        first_forecast = df['forecast_algo'].iloc[0]
        
        analyzer.compare_forecast_algorithms(df, first_cpd, first_delta, first_trend)
        analyzer.compare_trend_orders(df, first_cpd, first_delta, first_forecast)
        analyzer.compare_cpd_algorithms(df, first_delta, first_trend, first_forecast)
    
    # Generate plots
    if HAS_MATPLOTLIB:
        analyzer.plot_algorithm_comparison(df, 'rmse')
        analyzer.plot_algorithm_comparison(df, 'mae')
        analyzer.plot_trend_comparison(df, 'rmse')
    
    # Save full results to CSV
    output_file = script_dir / 'results' / 'forecast_results' / f"results_analysis_{args.dataset}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nFull results saved to: {output_file}")


if __name__ == '__main__':
    main()
