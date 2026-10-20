#!/bin/bash
# ==================================================
# SPIN-RNA Batch Training & Testing Script (23 Benchmarks)
# ==================================================

DATASETS=("h_b" "h_k" "h_l" "m_b" "m_h" "m_k" "m_l" "m_t" "r_b" "r_k" "r_l" "RMBase_h_PseudoU" "RMBase_h_m6A" "RMBase_m_m6A" "RMBase_r_m6A" "Atlas_h_m1A" "Atlas_m_m1A" "Atlas_h_m5C" "Atlas_m_m5C" "Atlas_h_m6A" "Atlas_m_m6A" "Atlas_r_m6A" "Atlas_h_m7G")
MODEL="pinn"
EPOCHS=30
BATCH_SIZE=32
LR="1e-3"

for DATASET in "${DATASETS[@]}"; do
    echo "[$(date)] >> Running SPIN-RNA on dataset: $DATASET"
    if [ -f "data-processed/train/benchmark/${DATASET}.fa" ]; then
        TRAIN_DATA="data-processed/train/benchmark/${DATASET}.fa"
    else
        TRAIN_DATA="data-processed/train/benchmark/${DATASET}_all.fa"
    fi
    TEST_DATA="data-processed/test/independent/${DATASET}_Test.fa"

    python3 run.py \
        --mode all \
        --model_type "$MODEL" \
        --train_data "$TRAIN_DATA" \
        --test_data "$TEST_DATA" \
        --epochs "$EPOCHS" \
        --k_folds 5 \
        --lr "$LR" \
        --batch_size "$BATCH_SIZE"
done
