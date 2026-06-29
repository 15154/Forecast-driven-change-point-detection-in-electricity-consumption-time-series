"""
Author: Mathias de Schietere
Organization: UCLouvain
GitHub: https://github.com/15154
Created: 2026-01-25
"""

# ECOS2026 - Energy Consumption Time Series Analysis Pipeline

A comprehensive Python pipeline for forecasting energy consumption and detecting change points in time series data. The pipeline is structured, modular, and follows programming standards for maintainability and reusability.

## Overview

The ECOS2026 pipeline performs a complete analysis workflow:

1. **Step 1: CPD on Original Data** - Detect change points in raw energy consumption data
2. **Step 2: Forecasting** - Forecast consumption using multiple machine learning methods
3. **Step 3: Residual Analysis** - Compute and analyze forecast residuals
4. **Step 4: CPD on Residuals** - Detect change points in residual data
5. **Step 5: CPD Comparison** - Compare CPD results with and without forecasting

## Directory Structure

```
ECOS2026/
├── pipeline/                      # Main pipeline modules
│   ├── __init__.py
│   ├── main.py                    # Main orchestrator
│   ├── data_loader.py             # Data loading utilities
│   ├── forecasting.py             # Forecasting engine
│   ├── cpd_pipeline.py            # Change point detection wrapper
│   ├── visualization.py           # Plotting and visualization
│   ├── step1_cpd_original.py      # Step 1 implementation
│   ├── step2_forecasting.py       # Step 2 implementation
│   ├── step3_residuals.py         # Step 3 implementation
│   ├── step4_cpd_residuals.py     # Step 4 implementation
│   └── step5_cpd_comparison.py    # Step 5 implementation
├── main.py                        # Python local launcher
├── README.md                      # This file
├── results/                       # Output results (created at runtime)
│   ├── {dataset_name}/
│   │   ├── step1_cpd_original/
│   │   │   ├── figures/
│   │   │   └── metrics/
│   │   ├── step2_forecasting/
│   │   │   ├── figures/
│   │   │   └── metrics/
│   │   ├── step3_residuals/
│   │   │   ├── figures/
│   │   │   └── metrics/
│   │   ├── step4_cpd_residuals/
│   │   │   ├── figures/
│   │   │   └── metrics/
│   │   └── step5_cpd_comparison/
│   │       └── metrics/
│   └── SUMMARY_REPORT.txt
└── logs/                          # Execution logs (created at runtime)
```

## Installation

### Prerequisites

- Python 3.8 or later
- Linux/Unix environment (tested on Linux)
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - statsmodels
  - matplotlib
  - seaborn
  - rpy2 (for R-based CPD methods)
  - ruptures (for change point detection)

### Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) For R-based CPD methods, ensure R is installed:
```bash
# Linux (Ubuntu/Debian)
sudo apt-get install r-base

# Check R installation
Rscript --version
```

3. Navigate to ECOS2026 directory:
```bash
cd ECOS2026
```

## Usage

### Local Execution with Python

Use `ECOS2026/main.py` to run the same worker logic used by SLURM jobs, without bash wrappers.

```bash
cd serenity/ECOS2026

# One local job (same structure as one SLURM worker job)
python3 main.py run-one \
  --dataset LUCID_1 \
  --cpd-algo BottomUp \
  --forecast-algo LR \
  --min-segment 28 --delta 1 --window-days 28

# Full local batch (sequential) for one dataset + one CPD algo
python3 main.py run-batch \
  --dataset LUCID_1 \
  --cpd-algo BottomUp

# Preview commands only
python3 main.py run-batch \
  --dataset ASHRAE901_ApartmentMidRise_STD2019 \
  --cpd-algo Window \
  --dry-run
```

### Paper Figure: Forecast-Driven vs Raw CPD (Mirrored Bars)

Generate the aggregated raw-vs-residual comparison tables first:

```bash
cd serenity
python3 ECOS2026/analyze_cpd_raw_vs_residuals.py
```

Then generate the mirrored chart (positive bars: residual/forecast-driven better, negative bars: raw better):

```bash
cd serenity
python3 ECOS2026/plot_forecast_driven_vs_raw_cpd.py --metric f1_score
```

Outputs:
- `ECOS2026/results/forecast_results/forecast_driven_vs_raw_cpd_counts_f1_score.csv`
- `LATEX/figures/plots/forecast_driven_vs_raw_cpd_mirrored_f1_score.png`

## Pipeline Steps

### Step 1: CPD on Original Data

Detects change points in the original energy consumption time series using multiple algorithms:
- **Algorithms**: BottomUp, Pelt, Binseg, Window, KernelCPD, CUSUM, EWMA, TwoSample, SegNeighMean, SegNeighVar
- **Output**:
  - `figures/01_original_data.png` - Original data with true change points
  - `figures/02_cpd_{algorithm}.png` - CPD results per algorithm
  - `metrics/grid_search_{algorithm}.csv` - Grid search parameters
  - `metrics/cpd_original_summary.csv` - Summary metrics

### Step 2: Forecasting

Forecasts year 3 consumption using multiple machine learning models:
- **Models**: LinearRegression, Lasso, KNN, SVR, XGBoost, MLP
- **Features**: 
  - Lagged consumption values
  - Polynomial features
  - Seasonal indicators (day of week, month, etc.)
  - Fourier terms for seasonality
  - Trend terms
- **Output**:
  - `figures/03_forecasts_comparison.png` - Comparison of all forecasts
  - `metrics/forecasting_metrics.csv` - MSE, MAPE, RMSE per model
  - `metrics/predictions.csv` - Actual vs predicted values

### Step 3: Residual Analysis

Computes and analyzes residuals (actual - predicted):
- **Metrics**: Mean, Std, Min, Max, Median of residuals
- **Output**:
  - `figures/04_residuals_{model}.png` - Time series and distribution plots
  - `metrics/residual_statistics.csv` - Statistical summary
  - `metrics/residuals_timeseries.csv` - Residual time series

### Step 4: CPD on Residuals

Detects change points in forecast residuals (with or without forecasts):
- **Purpose**: Identify if residuals have systematic changes
- **Output**:
  - `figures/05_residuals_{model}.png` - Residual plots with true CPs
  - `figures/06_cpd_residuals_{model}_{algorithm}.png` - CPD results
  - `metrics/cpd_residuals_summary.csv` - Performance metrics

### Step 5: CPD Comparison

Compares CPD performance on original data vs residuals:
- **Metrics**: Precision, Recall, F1-score, True/False Positives/Negatives
- **Analysis**: Improvement percentage, best performing combinations
- **Output**:
  - `metrics/cpd_comparison_{dataset}.csv` - Detailed comparison
  - `metrics/improvement_analysis_{dataset}.csv` - Improvement metrics

## Configuration

Modify settings in `pipeline/main.py`:

```python
# CPD algorithms to use
cpd_algorithms = ["BottomUp", "Pelt"]

# Forecasting models to use
forecast_models = ["LinearRegression", "Lasso", "KNN", "SVR", "XGBoost"]

# CPD parameters (in pipeline/cpd_pipeline.py)
min_segment = 10  # Minimum segment length
delta = timedelta(days=3)  # CP tolerance window
window_days = 20  # Sliding window size
```

## Output Structure

### Results Directory

Each dataset analysis creates:
```
results/{dataset_name}/
├── step1_cpd_original/
│   ├── figures/
│   │   ├── 01_original_data.png
│   │   └── 02_cpd_*.png
│   └── metrics/
│       ├── grid_search_*.csv
│       └── cpd_original_summary.csv
├── step2_forecasting/
│   ├── figures/
│   │   └── 03_forecasts_comparison.png
│   └── metrics/
│       ├── forecasting_metrics.csv
│       └── predictions.csv
├── step3_residuals/
│   ├── figures/
│   │   └── 04_residuals_*.png
│   └── metrics/
│       ├── residual_statistics.csv
│       └── residuals_timeseries.csv
├── step4_cpd_residuals/
│   ├── figures/
│   │   ├── 05_residuals_*.png
│   │   └── 06_cpd_residuals_*.png
│   └── metrics/
│       └── cpd_residuals_summary.csv
└── step5_cpd_comparison/
    └── metrics/
        ├── cpd_comparison_*.csv
        └── improvement_analysis_*.csv
```

### Logs

Execution logs are saved to `logs/pipeline_YYYYMMDD_HHMMSS.log`

## Module Documentation

### data_loader.py

`DataLoader` class for loading and preprocessing datasets.

```python
from pipeline.data_loader import DataLoader

loader = DataLoader("path/to/dataset.csv")
y_train, y_test, true_cps = loader.split_years(train_years=2)
```

### forecasting.py

`ForecastingEngine` class for training and making predictions.

```python
from pipeline.forecasting import ForecastingEngine

engine = ForecastingEngine(X_train, y_train)
engine.train_all_models()
predictions = engine.predict(X_test, model_name="XGBoost")
metrics = engine.evaluate(y_test, predictions)
```

### cpd_pipeline.py

`CPDPipeline` class for change point detection.

```python
from pipeline.cpd_pipeline import CPDPipeline

cpd = CPDPipeline(min_segment=10, delta=timedelta(days=3))
detected_cps = cpd.detect_sliding_window(y_test, estimator)
eval_results = cpd.evaluate(y_test, detected_cps, true_cps)
```

### visualization.py

`Visualizer` class for creating plots.

```python
from pipeline.visualization import Visualizer

viz = Visualizer(output_dir)
viz.plot_timeseries(y_test, title="Data", filename="plot.png")
viz.plot_forecast_comparison(y_test, predictions, filename="forecast.png")
viz.plot_cpd_results(y_test, true_cps, detected_cps, filename="cpd.png")
```

## Performance Considerations

- **Memory**: Pipeline requires ~2-4GB for large datasets
- **Runtime**: Full pipeline on single dataset: ~5-15 minutes (depending on data size and number of models)
- **Parallelization**: Set `parallel=True` in `main.py` for multi-dataset processing

## Troubleshooting

### Python Module Not Found
```bash
# Check Python path
python3 -c "import sys; print(sys.path)"

# Add to PYTHONPATH if needed
export PYTHONPATH="serenity:$PYTHONPATH"
```

### Missing R Dependencies
```bash
# Install required R packages
R -e "install.packages('changepoint')"
```

### Memory Issues
- Reduce number of models in configuration
- Process datasets sequentially (not in parallel)
- Increase available system memory or use smaller datasets

### CPD Algorithm Failures
- Adjust `min_segment` parameter (try 5-20)
- Adjust `window_days` for sliding window
- Check if time series has sufficient length (>50 points recommended)

## Contributing

To add new forecasting models:

1. Add method to `ForecastingEngine` class in `pipeline/forecasting.py`
2. Add to `train_all_models()` method
3. Update configuration in `pipeline/main.py`

To add new CPD algorithms:

1. Update `setup_param_grid()` in `pipeline/cpd_pipeline.py`
2. Ensure algorithm is available in CPDinterface from serenity

## References

### Datasets
- Located in `datasets/processed/profiles/`
- Format: CSV with columns [consumption, true_change_point, trend]

### Original Code
- CPD interface: `python/CPDinterface.py`
- Example notebooks: `serenity/notebooks/`

## License

This project is part of the Serenity research initiative.

---

**Last Updated**: March 13, 2026
