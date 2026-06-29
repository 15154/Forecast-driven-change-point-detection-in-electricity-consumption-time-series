#!/bin/bash

# """
# Author: Mathias de Schietere
# Organization: UCLouvain
# GitHub: https://github.com/15154
# Created: 2026-01-25
# """

## ECOS2026 SLURM Worker Submission Script
## Submits a single job with specific parameters
##
## Called by: slurm_ecos2026_loop.sh
##

# Parameters from loop script (one job per forecast algorithm; deterministic terms auto-selected)
DATASET="$1"
CPD_ALGO="$2"
MIN_SEG="$3"
DELTA="$4"
WINDOW="$5"
FORECAST_ALGO="$6"
ALGO_PARAMS="$7"
MAIL="${8}"
PARTITION="${9}"
TIME="${10}"
MEMORY="${11}"
SCRIPT_DIR="${12}"
PROJECT_ROOT="${13}"

# Create job name (no longer includes trend order since it's gridsearch)
JOB_NAME="ecos_${DATASET}_${CPD_ALGO}_ms${MIN_SEG}_d${DELTA}_w${WINDOW}_${FORECAST_ALGO}"

# Create output directory for logs
LOG_DIR="$SCRIPT_DIR/results/raw/slurm_logs"
mkdir -p "$LOG_DIR"

# Submit job
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$LOG_DIR/${JOB_NAME}_%j.log
#SBATCH --error=$LOG_DIR/${JOB_NAME}_%j.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=$TIME
#SBATCH --partition=$PARTITION
#SBATCH --mem=$MEMORY
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=$MAIL

# Load Python environment if needed
module load Python/3.10.4 2>/dev/null || true

# Activate virtual environment if it exists
if [[ -f "$PROJECT_ROOT/venv/bin/activate" ]]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Run the analysis with parameters (trend/fourier auto-selected in worker)
cd "$SCRIPT_DIR"
python3 slurm_ecos2026_worker.py \
    --dataset "$DATASET" \
    --cpd-algo "$CPD_ALGO" \
    --min-segment "$MIN_SEG" \
    --delta "$DELTA" \
    --window-days "$WINDOW" \
    --forecast-algo "$FORECAST_ALGO" \
    --algo-params "$ALGO_PARAMS" \
    --output-dir "$SCRIPT_DIR/results/raw" \
    --config "$SCRIPT_DIR/config_energyplus.json"
EOT
