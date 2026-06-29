#!/bin/bash

# """
# Author: Mathias de Schietere
# Organization: UCLouvain
# GitHub: https://github.com/15154
# Created: 2026-02-23
# Updated: 2026-02-23 - Unified loop script with dataset and algorithm parameters
# """

## ECOS2026 Unified SLURM Loop Script
## Submits CPD + forecasting jobs for a specific dataset and CPD algorithm
##
## Usage: sh slurm_ecos2026_loop.sh <DATASET> <CPD_ALGORITHM> [email]
## Examples:
##   sh slurm_ecos2026_loop.sh LUCID_1 Binseg
##   sh slurm_ecos2026_loop.sh LUCID_1 Binseg mathias.deschietere@uclouvain.be
##   sh slurm_ecos2026_loop.sh ASHRAE901_ApartmentMidRise_STD2019 Window
##   sh slurm_ecos2026_loop.sh ASHRAE901_OfficeSmall_STD2019 Pelt john@uclouvain.be
##
## Supported CPD Algorithms: Binseg, BottomUp, CUSUM, Pelt, KernelCPD, Window, Dynp, TwoSample, EWMA, WBS, cpm1B, cpm1S, cpmMS
##

set -euo pipefail

# ============================================================================
# COMMAND-LINE ARGUMENT PARSING
# ============================================================================

if [[ $# -lt 2 ]]; then
    echo "ERROR: Insufficient arguments"
    echo ""
    echo "Usage: sh slurm_ecos2026_loop.sh <DATASET> <CPD_ALGORITHM> [email]"
    echo ""
    echo "Arguments:"
    echo "  DATASET        - Dataset name (e.g., LUCID_1, ASHRAE901_ApartmentMidRise_STD2019)"
    echo "  CPD_ALGORITHM  - CPD algorithm (e.g., Binseg, BottomUp, Pelt, Window, KernelCPD, CUSUM, EWMA, WBS, Dynp, TwoSample, cpm1B, cpm1S, cpmMS)"
    echo "  email          - Optional email for SLURM notifications (default: mathias.deschietere@uclouvain.be)"
    echo ""
    echo "Examples:"
    echo "  sh slurm_ecos2026_loop.sh LUCID_1 Binseg"
    echo "  sh slurm_ecos2026_loop.sh ASHRAE901_ApartmentMidRise_STD2019 Window yourname@uclouvain.be"
    echo ""
    exit 1
fi

DATASET="$1"
CPD_ALGO="$2"
MAIL="${3:-mathias.deschietere@uclouvain.be}"

# Normalize dataset name (remove .csv extension if provided)
DATASET="${DATASET%.csv}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================="
echo "ECOS2026 Unified SLURM Loop"
echo "=================================="
echo "Dataset: $DATASET"
echo "CPD Algorithm: $CPD_ALGO"
echo "Email: $MAIL"
echo "Project directory: $PROJECT_ROOT"
echo ""

# ============================================================================
# DATASET VALIDATION
# ============================================================================

DATASET_DIR="$PROJECT_ROOT/datasets/processed/profiles"
DATASET_FILE="$DATASET_DIR/${DATASET}.csv"

if [[ ! -f "$DATASET_FILE" ]]; then
    echo "ERROR: Dataset not found: $DATASET_FILE"
    echo ""
    echo "Available datasets:"
    ls "$DATASET_DIR"/*.csv 2>/dev/null | xargs -n1 basename | sed 's/\.csv$//' | sed 's/^/  /'
    echo ""
    exit 1
fi

echo "✓ Dataset found: $DATASET_FILE"

# Determine if LUCID or EnergyPlus
if [[ "$DATASET" == LUCID* ]]; then
    DATA_SOURCE="LUCID"
    echo "✓ Data source: LUCID + Open-Meteo"
elif [[ "$DATASET" == ASHRAE901* ]]; then
    DATA_SOURCE="ENERGYPLUS"
    echo "✓ Data source: EnergyPlus"
else
    echo "⚠ Data source: Unknown (will default to LUCID if enabled in CONFIG.yml)"
    DATA_SOURCE="UNKNOWN"
fi

echo ""

# ============================================================================
# CLUSTER DETECTION & SLURM RESOURCES
# ============================================================================

hostname=$(hostname)
echo "Detected cluster: $hostname"

if [[ $hostname == dragon1* ]]; then
    PARTITION=long
    TIME="41-00:00:0"
    MEMORY=128G
elif [[ $hostname == dragon2* ]]; then
    PARTITION=long
    TIME="21-00:00:0"
    MEMORY=192G
elif [[ $hostname == hercules* ]]; then
    PARTITION=batch
    TIME="15-00:00:0"
    MEMORY=256G
elif [[ $hostname == nic5* ]]; then
    PARTITION=batch
    TIME="2-00:00:0"
    MEMORY=256G
elif [[ $hostname == lyra* ]]; then
    PARTITION=batch
    TIME="5-00:00:0"
    MEMORY=128G
elif [[ $hostname == lemaitre4* ]]; then
    PARTITION=batch
    TIME="2-00:00:0"
    MEMORY=128G
else
    PARTITION=batch
    TIME="2-00:00:00"
    MEMORY=64G
fi

echo "SLURM Configuration:"
echo "  Partition: $PARTITION"
echo "  Time: $TIME"
echo "  Memory: $MEMORY"
echo ""

# ============================================================================
# PARAMETER RANGES
# ============================================================================

# CPD Parameters (consistent across all algorithms)
MIN_SEGMENTS=(28)
DELTAS=(1 3 7)
WINDOW_DAYS=(28)

# Feature engineering is now automatic in worker.py:
# - trend degree selected analytically (nested F-tests)
# - fourier order selected analytically (periodogram)
# - weekly seasonal flag selected analytically (periodogram)
# - lags selected via PACF

# Forecasting Algorithms - one job per algorithm
FORECAST_ALGOS=("LR" "Lasso" "KNN" "SVR" "LSVR" "SGD" "MLP" "XGB" "ARIMA" "SARIMAX" "HybridLassoXGB" "HybridLRXGB" "HybridLassoMLP" "HybridKNNMLP")

# Algorithm-specific Parameters
LASSO_ALPHAS=(0.1 1.0 10.0 100.0 1000.0)
KNN_NEIGHBORS=(3 5 7 10)
SVR_KERNELS=("rbf" "linear" "poly")
SVR_C_VALUES=(0.1 1.0 10.0 100.0 1000.0)
LSVR_C_VALUES=(0.1 1.0 10.0 100.0 1000.0)
SGD_LEARNING_RATES=("optimal" "constant" "invscaling")
SGD_ETA0_VALUES=(0.001 0.01 0.1)
MLP_HIDDEN_LAYERS=("100" "100,50" "100,50,25" "200,100")
XGB_DEPTHS=(2 4 6 8 10)
XGB_ESTIMATORS=(50 100 200 500 100)
XGB_LEARNING_RATES=(0.05 0.10 0.20 0.30 0.50)
ARIMA_ORDERS=("1,1,1" "2,1,1" "1,1,2" "2,1,2")
SARIMAX_ORDERS=("1,1,1" "2,1,1" "1,1,2")
SARIMAX_SEASONAL_ORDERS=("1,1,1,12" "2,1,1,12")

# Hybrid Estimators
HYBRID_LASSO_ALPHAS=(0.1 1.0 10.0)
HYBRID_XGB_DEPTHS=(5 7)
HYBRID_XGB_ESTIMATORS=(100 200)
HYBRID_XGB_LR=(0.1 0.2)
HYBRID_MLP_HIDDEN=("64,32" "128,64")
HYBRID_MLP_LR=(0.001 0.01)
HYBRID_KNN_NEIGHBORS=(5 7)

# ============================================================================
# JOB CALCULATION
# ============================================================================

echo "Parameter Configuration:"
echo "  Feature Engineering (automatic, no gridsearch):"
echo "    Trend degree: auto-selected"
echo "    Fourier order: auto-selected"
echo "    Weekly seasonal flag: auto-selected"
echo "    Lags: PACF-selected"
echo ""
echo "  CPD Algorithm: $CPD_ALGO"
echo "  CPD Parameters:"
echo "    MIN_SEGMENT values: ${#MIN_SEGMENTS[@]} (${MIN_SEGMENTS[*]})"
echo "    DELTA values: ${#DELTAS[@]} (${DELTAS[*]})"
echo "    WINDOW_DAYS values: ${#WINDOW_DAYS[@]} (${WINDOW_DAYS[*]})"
echo ""
echo "  Forecasting:"
echo "    Algorithms: ${#FORECAST_ALGOS[@]} (${FORECAST_ALGOS[*]})"
echo ""
echo ""
# Calculate total jobs
TOTAL_JOBS=$((${#MIN_SEGMENTS[@]} * ${#DELTAS[@]} * ${#WINDOW_DAYS[@]} * ${#FORECAST_ALGOS[@]}))
echo "Total Jobs to Submit:"
echo "  ${#MIN_SEGMENTS[@]} × ${#DELTAS[@]} × ${#WINDOW_DAYS[@]} × ${#FORECAST_ALGOS[@]} = $TOTAL_JOBS"
echo ""

# ============================================================================
# USER CONFIRMATION
# ============================================================================

echo " This will submit $TOTAL_JOBS SLURM jobs for:"
echo "  Dataset: $DATASET"
echo "  Algorithm: $CPD_ALGO"
echo "  Email notifications to: $MAIL"
echo ""

read -p "Proceed with job submission? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Starting job submissions..."
echo "=================================="
echo ""

# ============================================================================
# MAIN LOOP: SUBMIT ALL COMBINATIONS
# ============================================================================

JOB_COUNT=0

for MIN_SEG in "${MIN_SEGMENTS[@]}"; do
    for DELTA in "${DELTAS[@]}"; do
        for WINDOW in "${WINDOW_DAYS[@]}"; do
            # Skip invalid combinations (window should be >= min_segment)
            if [[ $WINDOW -lt $MIN_SEG ]]; then
                continue
            fi

            for FORECAST_ALGO in "${FORECAST_ALGOS[@]}"; do
                # Build algorithm-specific parameters based on FORECAST_ALGO
                ALGO_PARAMS=""
                
                case "$FORECAST_ALGO" in
                    "LR")
                        ALGO_PARAMS=""
                        ;;
                    "Lasso")
                        ALGO_PARAMS="$(IFS=,; echo "${LASSO_ALPHAS[*]}")"
                        ;;
                    "KNN")
                        ALGO_PARAMS="$(IFS=,; echo "${KNN_NEIGHBORS[*]}")"
                        ;;
                    "SVR")
                        SVR_KERNELS_STR="$(IFS=,; echo "${SVR_KERNELS[*]}")"
                        SVR_C_STR="$(IFS=,; echo "${SVR_C_VALUES[*]}")"
                        ALGO_PARAMS="$SVR_KERNELS_STR:$SVR_C_STR"
                        ;;
                    "LSVR")
                        ALGO_PARAMS="$(IFS=,; echo "${LSVR_C_VALUES[*]}")"
                        ;;
                    "SGD")
                        SGD_LR_STR="$(IFS=,; echo "${SGD_LEARNING_RATES[*]}")"
                        SGD_ETA_STR="$(IFS=,; echo "${SGD_ETA0_VALUES[*]}")"
                        ALGO_PARAMS="$SGD_LR_STR:$SGD_ETA_STR"
                        ;;
                    "MLP")
                        ALGO_PARAMS="$(IFS=,; echo "${MLP_HIDDEN_LAYERS[*]}")"
                        ;;
                    "XGB")
                        XGB_D_STR="$(IFS=,; echo "${XGB_DEPTHS[*]}")"
                        XGB_N_STR="$(IFS=,; echo "${XGB_ESTIMATORS[*]}")"
                        XGB_LR_STR="$(IFS=,; echo "${XGB_LEARNING_RATES[*]}")"
                        ALGO_PARAMS="$XGB_D_STR:$XGB_N_STR:$XGB_LR_STR"
                        ;;
                    "ARIMA")
                        ALGO_PARAMS="$(IFS=,; echo "${ARIMA_ORDERS[*]}")"
                        ;;
                    "SARIMAX")
                        ORDERS_STR="$(IFS=,; echo "${SARIMAX_ORDERS[*]}")"
                        SEASONAL_STR="$(IFS=,; echo "${SARIMAX_SEASONAL_ORDERS[*]}")"
                        ALGO_PARAMS="$ORDERS_STR:$SEASONAL_STR"
                        ;;
                    "HybridLassoXGB")
                        LASSO_A_STR="$(IFS=,; echo "${HYBRID_LASSO_ALPHAS[*]}")"
                        XGB_D_STR="$(IFS=,; echo "${HYBRID_XGB_DEPTHS[*]}")"
                        XGB_N_STR="$(IFS=,; echo "${HYBRID_XGB_ESTIMATORS[*]}")"
                        XGB_LR_STR="$(IFS=,; echo "${HYBRID_XGB_LR[*]}")"
                        ALGO_PARAMS="$LASSO_A_STR:$XGB_D_STR:$XGB_N_STR:$XGB_LR_STR"
                        ;;
                    "HybridLRXGB")
                        XGB_D_STR="$(IFS=,; echo "${HYBRID_XGB_DEPTHS[*]}")"
                        XGB_N_STR="$(IFS=,; echo "${HYBRID_XGB_ESTIMATORS[*]}")"
                        XGB_LR_STR="$(IFS=,; echo "${HYBRID_XGB_LR[*]}")"
                        ALGO_PARAMS="$XGB_D_STR:$XGB_N_STR:$XGB_LR_STR"
                        ;;
                    "HybridLassoMLP")
                        LASSO_A_STR="$(IFS=,; echo "${HYBRID_LASSO_ALPHAS[*]}")"
                        MLP_H_STR="$(IFS=,; echo "${HYBRID_MLP_HIDDEN[*]}")"
                        MLP_LR_STR="$(IFS=,; echo "${HYBRID_MLP_LR[*]}")"
                        ALGO_PARAMS="$LASSO_A_STR:$MLP_H_STR:$MLP_LR_STR"
                        ;;
                    "HybridKNNMLP")
                        KNN_N_STR="$(IFS=,; echo "${HYBRID_KNN_NEIGHBORS[*]}")"
                        MLP_H_STR="$(IFS=,; echo "${HYBRID_MLP_HIDDEN[*]}")"
                        MLP_LR_STR="$(IFS=,; echo "${HYBRID_MLP_LR[*]}")"
                        ALGO_PARAMS="$KNN_N_STR:$MLP_H_STR:$MLP_LR_STR"
                        ;;
                esac

                JOB_COUNT=$((JOB_COUNT + 1))

                # Submit job: one per (cpd_params, forecast_algo)
                sh "$SCRIPT_DIR/slurm_ecos2026_submit.sh" \
                    "$DATASET" "$CPD_ALGO" \
                    "$MIN_SEG" "$DELTA" "$WINDOW" \
                    "$FORECAST_ALGO" "$ALGO_PARAMS" \
                    "$MAIL" "$PARTITION" "$TIME" "$MEMORY" \
                    "$SCRIPT_DIR" "$PROJECT_ROOT"

                # Print progress every 10 jobs
                if (( JOB_COUNT % 10 == 0 )); then
                    echo "  Submitted $JOB_COUNT/$TOTAL_JOBS jobs..."
                fi
            done
        done
    done
done

echo ""
echo "=================================="
echo "✓ Job Submission Complete!"
echo "=================================="
echo "Dataset: $DATASET"
echo "Algorithm: $CPD_ALGO"
echo "Total jobs submitted: $JOB_COUNT"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  watch -n 5 'squeue -u \$USER | head -20'"
echo ""
echo "View job status:"
echo "  sacct -u \$USER"
echo ""
echo "Cancel all jobs if needed:"
echo "  scancel -u \$USER"
echo ""
