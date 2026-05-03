#!/bin/bash
#
# Ablation study for hierarchy weight schemes
#
# This script runs 3 training experiments with different weight schemes:
# 1. Uniform: All levels weighted equally (1.0)
# 2. Linear: Weights increase linearly with depth (1,2,3,4,5,6,7)
# 3. Piecewise: Current scheme with steeper slope for genus/species (0.5-2.5)
#
# Each experiment uses 5-fold cross-validation for robust comparison.
#
# Usage:
#   ./scripts/run_weight_ablation.sh
#
# Or run individual experiments:
#   ./scripts/run_weight_ablation.sh uniform
#   ./scripts/run_weight_ablation.sh linear
#   ./scripts/run_weight_ablation.sh piecewise

set -e

cd /mnt/beegfs/home/dzimmerman2021/CAP6415_F25_project-Kaggle-FathomNet

PYTHON=".venv/bin/python"
CONFIG="config/experiment-multiscale.yaml"
SCALES="1x 3x 5x"
N_FOLDS=5
ALPHA=0.3

run_experiment() {
    local scheme=$1
    local output_dir="outputs/ablation_weights_${scheme}"

    echo "============================================================"
    echo "Running weight scheme: ${scheme}"
    echo "Output: ${output_dir}"
    echo "============================================================"

    $PYTHON -m src.training.train_kfold \
        --config $CONFIG \
        --n-folds $N_FOLDS \
        --scales $SCALES \
        --output-dir $output_dir \
        --loss-type distance \
        --alpha $ALPHA \
        --weight-scheme $scheme \
        --seed 42

    echo ""
    echo "Completed: ${scheme}"
    echo "Results saved to: ${output_dir}"
    echo ""
}

# Run specific experiment if argument provided
if [ $# -eq 1 ]; then
    run_experiment $1
    exit 0
fi

# Run all experiments
echo "============================================================"
echo "HIERARCHY WEIGHT ABLATION STUDY"
echo "============================================================"
echo ""
echo "Weight schemes:"
echo "  uniform:   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]"
echo "  linear:    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]"
echo "  piecewise: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]"
echo ""
echo "Each experiment uses ${N_FOLDS}-fold cross-validation"
echo "============================================================"
echo ""

# Run experiments sequentially (each is GPU-intensive)
for scheme in uniform linear piecewise; do
    run_experiment $scheme
done

echo "============================================================"
echo "ABLATION STUDY COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  outputs/ablation_weights_uniform/"
echo "  outputs/ablation_weights_linear/"
echo "  outputs/ablation_weights_piecewise/"
echo ""
echo "To generate submissions and compare:"
echo "  python -m src.inference.ensemble_predict --checkpoint-dir outputs/ablation_weights_uniform"
echo "  python -m src.inference.ensemble_predict --checkpoint-dir outputs/ablation_weights_linear"
echo "  python -m src.inference.ensemble_predict --checkpoint-dir outputs/ablation_weights_piecewise"
