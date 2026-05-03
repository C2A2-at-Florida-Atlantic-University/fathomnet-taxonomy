#!/bin/bash
# HXE vs Expected Distance Ablation Study
# Compares Hierarchical Cross-Entropy (Bertinetto et al. 2020) with
# Expected Taxonomic Distance approach

set -e

cd /mnt/beegfs/home/dzimmerman2021/CAP6415_F25_project-Kaggle-FathomNet

# Use the project's virtual environment
PYTHON=".venv/bin/python"

# Configuration
CONFIG="config/experiment-multiscale.yaml"
N_FOLDS=5
ALPHA=0.3  # Balance between CE and hierarchical term
SCALES="1x 3x 5x"
BASE_OUTPUT="outputs/ablation_hxe"

echo "=========================================="
echo "HXE vs Expected Distance Ablation Study"
echo "=========================================="
echo "Config: $CONFIG"
echo "Folds: $N_FOLDS"
echo "Alpha: $ALPHA"
echo "Scales: $SCALES"
echo "=========================================="

# Create output directory
mkdir -p $BASE_OUTPUT

# Experiment 1: Expected Distance (our approach)
echo ""
echo "[1/2] Running Expected Distance approach..."
$PYTHON -m src.training.train_kfold \
    --config $CONFIG \
    --n-folds $N_FOLDS \
    --loss-type distance \
    --alpha $ALPHA \
    --scales $SCALES \
    --output-dir "${BASE_OUTPUT}/expected_distance" \
    2>&1 | tee "${BASE_OUTPUT}/expected_distance.log"

# Experiment 2: HXE (Bertinetto et al.)
echo ""
echo "[2/2] Running HXE approach (Bertinetto et al.)..."
$PYTHON -m src.training.train_kfold \
    --config $CONFIG \
    --n-folds $N_FOLDS \
    --loss-type hxe \
    --alpha $ALPHA \
    --scales $SCALES \
    --output-dir "${BASE_OUTPUT}/hxe" \
    2>&1 | tee "${BASE_OUTPUT}/hxe.log"

echo ""
echo "=========================================="
echo "Ablation Complete!"
echo "=========================================="
echo "Results saved to: $BASE_OUTPUT"
echo ""
echo "Expected Distance: ${BASE_OUTPUT}/expected_distance/"
echo "HXE:               ${BASE_OUTPUT}/hxe/"
echo ""

# Extract and compare final results
echo "Summary of validation scores:"
echo "-----------------------------"
for exp in expected_distance hxe; do
    echo "$exp:"
    grep -h "val_tax_score" "${BASE_OUTPUT}/${exp}/fold_*/logs/fold_*/version_*/metrics.csv" 2>/dev/null | tail -1 || \
    echo "  (Check TensorBoard logs for results)"
done
