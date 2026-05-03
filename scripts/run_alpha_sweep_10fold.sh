#!/bin/bash
# Alpha sensitivity sweep: Train 10-fold ensembles for multiple alpha values.
# Alpha=0.3 already exists at kfold_10fold_no_parent_cond/
# This script trains alpha in {0.0, 0.1, 0.2, 0.5}
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_alpha_sweep_10fold.sh
# Estimated time: ~40 GPU-hours total (4 alpha values × ~10 hrs each)

set -e

PYTHON=".venv/bin/python"
BASE_OUTPUT="/raid/scratch/dzimmerman2021/fathomnet/outputs"

# All settings match the best no-parent-cond config
COMMON_ARGS="--config config/experiment-multiscale.yaml \
    --n-folds 10 \
    --scales 1x 3x 5x full \
    --no-parent-conditioning \
    --devices 1 \
    --weight-scheme piecewise \
    --seed 42"

for ALPHA in 0.0 0.1 0.2 0.5; do
    OUTPUT_DIR="${BASE_OUTPUT}/kfold_10fold_alpha_${ALPHA}"

    # Skip if already complete (all 10 folds have checkpoints)
    EXISTING=$(find "${OUTPUT_DIR}" -name "*.ckpt" 2>/dev/null | wc -l)
    if [ "$EXISTING" -ge 10 ]; then
        echo "Alpha=${ALPHA}: Already have ${EXISTING} checkpoints, skipping."
        continue
    fi

    echo ""
    echo "============================================"
    echo "Training alpha=${ALPHA} (10-fold)"
    echo "============================================"

    $PYTHON src/training/train_kfold.py \
        $COMMON_ARGS \
        --alpha ${ALPHA} \
        --output-dir ${OUTPUT_DIR}

    echo ""
    echo "Evaluating alpha=${ALPHA} ensemble..."
    $PYTHON src/inference/ensemble_predict.py \
        --checkpoints ${OUTPUT_DIR}/fold_*/checkpoints/*.ckpt \
        --no-parent-conditioning \
        --output submission_10fold_alpha_${ALPHA}.csv \
        --decision min_risk \
        --tta hflip \
        --evaluate

    echo ""
    echo "Alpha=${ALPHA} complete."
done

echo ""
echo "============================================"
echo "Alpha sweep complete!"
echo "============================================"
echo ""
echo "Now run bootstrap CIs:"
echo "  python scripts/bootstrap_confidence.py \\"
echo "    --submissions submission_10fold_alpha_*.csv \\"
echo "               submission_no_parent_cond_10fold.csv \\"
echo "    --solution solution_kaggle_2025.csv \\"
echo "    --latex paper/tables/alpha_ci.tex"
echo ""
echo "And plot sensitivity curve:"
echo "  python scripts/plot_alpha_sensitivity.py"
