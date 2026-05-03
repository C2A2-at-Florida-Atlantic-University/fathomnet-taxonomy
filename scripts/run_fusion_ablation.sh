#!/bin/bash
#
# Fusion Method Ablation Study
#
# Compares different multi-scale fusion strategies:
# 1. concat:        Original concatenation (baseline)
# 2. hierarchy_attn: Per-level attention over scales (Option 2)
# 3. gated_cross:   Cross-scale self-attention (Option 3)
#
# Key hypothesis: hierarchy_attn should improve because:
# - Species level can focus on fine details (1x ROI)
# - Kingdom/Phylum can use more context (3x, 5x, full)
# - Each level learns optimal scale weighting
#
# Uses all 4 scales: 1x (ROI), 3x, 5x, full
#

set -e

cd /mnt/beegfs/home/dzimmerman2021/CAP6415_F25_project-Kaggle-FathomNet

PYTHON=".venv/bin/python"
CONFIG="config/experiment-multiscale.yaml"
SCALES="1x 3x 5x full"  # All 4 scales
N_FOLDS=5
ALPHA=0.3
LOSS_TYPE="distance"  # Our best loss function
BASE_OUTPUT="outputs/ablation_fusion"

echo "============================================================"
echo "FUSION METHOD ABLATION STUDY"
echo "============================================================"
echo ""
echo "Fusion methods:"
echo "  concat:        Concatenate all scale features (baseline)"
echo "  hierarchy_attn: Per-level attention over scales"
echo "  gated_cross:   Self-attention between scales"
echo ""
echo "Scales: $SCALES"
echo "Folds: $N_FOLDS"
echo "Loss: $LOSS_TYPE (alpha=$ALPHA)"
echo "============================================================"
echo ""

mkdir -p $BASE_OUTPUT

run_experiment() {
    local fusion=$1
    local output_dir="${BASE_OUTPUT}/${fusion}"

    echo "============================================================"
    echo "Running fusion method: ${fusion}"
    echo "Output: ${output_dir}"
    echo "============================================================"

    $PYTHON -m src.training.train_kfold \
        --config $CONFIG \
        --n-folds $N_FOLDS \
        --scales $SCALES \
        --output-dir $output_dir \
        --loss-type $LOSS_TYPE \
        --alpha $ALPHA \
        --fusion-type $fusion \
        --seed 42 \
        2>&1 | tee "${output_dir}.log"

    echo ""
    echo "Completed: ${fusion}"
    echo "Results saved to: ${output_dir}"
    echo ""
}

# Run specific experiment if argument provided
if [ $# -eq 1 ]; then
    run_experiment $1
    exit 0
fi

# Run all experiments sequentially (each is GPU-intensive)
for fusion in concat hierarchy_attn gated_cross; do
    run_experiment $fusion
done

echo "============================================================"
echo "ABLATION STUDY COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  ${BASE_OUTPUT}/concat/"
echo "  ${BASE_OUTPUT}/hierarchy_attn/"
echo "  ${BASE_OUTPUT}/gated_cross/"
echo ""
echo "To generate submissions and compare:"
echo "  python -m src.inference.ensemble_predict --checkpoint-dir ${BASE_OUTPUT}/concat"
echo "  python -m src.inference.ensemble_predict --checkpoint-dir ${BASE_OUTPUT}/hierarchy_attn"
echo "  python -m src.inference.ensemble_predict --checkpoint-dir ${BASE_OUTPUT}/gated_cross"
echo ""
echo "To analyze attention weights (hierarchy_attn only):"
echo "  python -c \"from scripts.analyze_fusion_attention import analyze; analyze('${BASE_OUTPUT}/hierarchy_attn')\""
