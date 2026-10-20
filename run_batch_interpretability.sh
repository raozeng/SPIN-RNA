#!/bin/bash
# ==================================================
# SPIN-RNA Batch Interpretability Script (23 Benchmarks)
# ==================================================

DATASETS=("h_b" "h_k" "h_l" "m_b" "m_h" "m_k" "m_l" "m_t" "r_b" "r_k" "r_l" "RMBase_h_PseudoU" "RMBase_h_m6A" "RMBase_m_m6A" "RMBase_r_m6A" "Atlas_h_m1A" "Atlas_m_m1A" "Atlas_h_m5C" "Atlas_m_m5C" "Atlas_h_m6A" "Atlas_m_m6A" "Atlas_r_m6A" "Atlas_h_m7G")
MODEL="pinn"

for DATASET in "${DATASETS[@]}"; do
    echo "[$(date)] >> Running Interpretability Suite for: $DATASET"
    python3 visualize_features.py --dataset "$DATASET" --model_type "$MODEL" --out_dir "result_inter/features"
    python3 visualize_saliency.py --dataset "$DATASET" --model_type "$MODEL" --top_n 3 --out_dir "result_inter/saliency"
    python3 visualize_confidence.py --dataset "$DATASET" --model_type "$MODEL" --out_dir "result_inter/confidence"
    python3 visualize_structure.py --dataset "$DATASET" --out_dir "result_inter/structure"
done
