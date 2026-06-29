#!/bin/bash

# Generate individual loop scripts for each CPD algorithm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CPD Algorithms
CPD_ALGOS=(
    "Pelt"
    "Binseg"
    "BottomUp"
    "Window"
    "KernelCPD"
    "CUSUM"
    "EWMA"
    "TwoSample"
    "sbs"
    "wbs"
    "cpm1B"
    "cpm1S"
    "cpmMS"
)

echo "Generating individual SLURM loop scripts for each CPD algorithm..."
echo ""

for CPD_ALGO in "${CPD_ALGOS[@]}"; do
    echo "Creating slurm_ecos2026_loop_${CPD_ALGO}.sh..."
    
    # Read the template (Pelt)
    TEMPLATE="$SCRIPT_DIR/slurm_ecos2026_loop_Pelt.sh"
    
    if [[ $CPD_ALGO != "Pelt" ]]; then
        OUTPUT="$SCRIPT_DIR/slurm_ecos2026_loop_${CPD_ALGO}.sh"
        
        # Copy template and replace CPD_ALGO
        sed "s/CPD_ALGO=\"Pelt\"/CPD_ALGO=\"${CPD_ALGO}\"/g; \
             s/ECOS2026 SLURM CPD: Pelt/ECOS2026 SLURM CPD: ${CPD_ALGO}/g; \
             s/loop_Pelt.sh/loop_${CPD_ALGO}.sh/g" \
            "$TEMPLATE" > "$OUTPUT"
        
        chmod +x "$OUTPUT"
        echo "  ✓ Created $OUTPUT"
    else
        chmod +x "$TEMPLATE"
        echo "  ✓ $TEMPLATE already exists"
    fi
done

echo ""
echo "Done! Generated ${#CPD_ALGOS[@]} scripts:"
for CPD_ALGO in "${CPD_ALGOS[@]}"; do
    echo "  - slurm_ecos2026_loop_${CPD_ALGO}.sh"
done
echo ""
echo "Usage: Submit each job to a different cluster:"
echo "  Cluster 1: bash slurm_ecos2026_loop_Pelt.sh"
echo "  Cluster 2: bash slurm_ecos2026_loop_Binseg.sh"
echo "  Cluster 3: bash slurm_ecos2026_loop_BottomUp.sh"
echo "  ... and so on"
echo ""
