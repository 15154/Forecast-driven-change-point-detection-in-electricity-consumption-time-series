#!/bin/bash

# """
# Author: Mathias de Schietere
# Organization: UCLouvain
# GitHub: https://github.com/15154
# Created: 2026-01-25
# """

## ECOS2026 Job Submission - Alternative Simplified Version
## 
## Use this if you want more control over which parameters to test
##
## Usage:
##   sh slurm_ecos2026_custom.sh dataset min_seg delta window lags order fourier
##
## Example:
##   sh slurm_ecos2026_custom.sh ASHRAE901_ApartmentMidRise_STD2019 10 3 20 15 2 8
##

if [[ $# -lt 7 ]]; then
    cat << 'EOF'
ECOS2026 Custom Job Submission

Usage: sh slurm_ecos2026_custom.sh [OPTIONS] dataset min_seg delta window lags order fourier

Positional Arguments:
  dataset          Dataset name (e.g., ASHRAE901_ApartmentMidRise_STD2019)
  min_seg          CPD minimum segment length
  delta            CPD tolerance in days
  window           CPD window size in days
  lags             Number of lagged features
  order            Polynomial order
  fourier          Fourier order

Optional Arguments:
  --xgb-depth D          XGBoost max depth (default: 10)
  --xgb-estimators N     XGBoost estimators (default: 100)
  --xgb-lr RATE          XGBoost learning rate (default: 0.1)
  --svr-c C              SVR C parameter (default: 10000)
  --svr-kernel K         SVR kernel (default: rbf)
  --knn-k K              KNN neighbors (default: 5)
  --email EMAIL          Email for SLURM notifications
  --time TIME            Max run time (default: 2-00:00:0)
  --memory MEM           Memory allocation (default: 64G)
  --cpus N               CPU count (default: 2)
  --partition PART       SLURM partition (default: batch)

Examples:
  # Basic submission
  sh slurm_ecos2026_custom.sh ASHRAE901_ApartmentMidRise_STD2019 10 3 20 15 2 8

  # With custom email
  sh slurm_ecos2026_custom.sh ASHRAE901_ApartmentMidRise_STD2019 10 3 20 15 2 8 \
      --email mathias.deschietere@uclouvain.be

  # Full customization
  sh slurm_ecos2026_custom.sh ASHRAE901_ApartmentMidRise_STD2019 15 7 30 31 4 12 \
      --xgb-depth 13 \
      --xgb-estimators 200 \
      --xgb-lr 0.15 \
      --svr-c 100000 \
      --svr-kernel poly \
      --knn-k 7 \
      --email mathias.deschietere@uclouvain.be \
      --time 5-00:00:0 \
      --memory 128G
EOF
    exit 1
fi

# Parse positional arguments
DATASET="$1"
MIN_SEG="$2"
DELTA="$3"
WINDOW="$4"
LAGS="$5"
ORDER="$6"
FOURIER="$7"

# Shift positional arguments and parse options
shift 7

# Defaults
XGB_D=10
XGB_N=100
XGB_LR=0.10
SVR_C=10000
SVR_K="rbf"
KNN_K=5
MAIL="${USER}@uclouvain.be"
TIME="2-00:00:0"
MEMORY="64G"
CPUS=2
PARTITION="batch"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --xgb-depth)
            XGB_D="$2"
            shift 2
            ;;
        --xgb-estimators)
            XGB_N="$2"
            shift 2
            ;;
        --xgb-lr)
            XGB_LR="$2"
            shift 2
            ;;
        --svr-c)
            SVR_C="$2"
            shift 2
            ;;
        --svr-kernel)
            SVR_K="$2"
            shift 2
            ;;
        --knn-k)
            KNN_K="$2"
            shift 2
            ;;
        --email)
            MAIL="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "ECOS2026 Custom Job Submission"
echo "========================================"
echo "Dataset: $DATASET"
echo "CPD: min_seg=$MIN_SEG, delta=$DELTA, window=$WINDOW"
echo "Forecasting: lags=$LAGS, order=$ORDER, fourier=$FOURIER"
echo "XGBoost: depth=$XGB_D, estimators=$XGB_N, lr=$XGB_LR"
echo "SVR: C=$SVR_C, kernel=$SVR_K"
echo "KNN: neighbors=$KNN_K"
echo "Email: $MAIL"
echo "Time: $TIME, Memory: $MEMORY, CPUs: $CPUS"
echo "========================================"
echo ""

# Submit job
sh "$SCRIPT_DIR/slurm_ecos2026_submit.sh" \
    "$DATASET" \
    "$MIN_SEG" "$DELTA" "$WINDOW" \
    "$LAGS" "$ORDER" "$FOURIER" \
    "$XGB_D" "$XGB_N" "$XGB_LR" \
    "$SVR_C" "$SVR_K" \
    "$KNN_K" \
    "$MAIL" "$PARTITION" "$TIME" "$MEMORY" "$CPUS" \
    "$SCRIPT_DIR" "$PROJECT_ROOT"

echo ""
echo "Job submitted! Monitor with:"
echo "  squeue -u \$USER -n 'ecos*'"
echo "  sacct -u \$USER -n 'ecos*' --format=JobID,JobName,State,Elapsed"
