# SPIN-RNA: A Structure-Aware, Physics-Informed Ensemble Framework for the Interpretable Prediction of Pan-RNA Modifications

This package contains the official PyTorch implementation, raw and ViennaRNA-folded benchmark datasets (`data/` and `data-processed/`), 5-fold cross-validation metric summaries (`train_model/`), independent test results (`test_result/`), and multi-view interpretability scripts and outputs (`result_inter/`) of **SPIN-RNA** across 23 mammalian benchmark datasets spanning three species (*Homo sapiens*, *Mus musculus*, and *Rattus norvegicus*) and five modification types (m6A, m1A, m5C, m7G, and Pseudouridine).

## 1. Package Structure

```text
SPIN-RNA_Code/
├── process_data.py               # Secondary-structure folding and MFE extraction via ViennaRNA
├── dataset.py                    # Sequence one-hot encoding, MFE adjacency graph, and biophysical descriptors
├── model.py                      # Neural architectures: SPIN-RNA (PINN), RNA-GCN, and Motif-CNN
├── train.py                      # Stratified 5-fold cross-validation training and metric logging
├── test.py                       # 5-fold ensemble evaluation on held-out independent test sets
├── run.py                        # Unified CLI entry point for training and independent testing
├── visualize_features.py         # Latent space projection (PCA and t-SNE) on independent test sets
├── visualize_confidence.py       # Prediction probability KDE and confidence calibration analysis
├── visualize_structure.py        # Secondary-structure spatial graph topology visualization
├── visualize_saliency.py         # Nucleotide-level input-gradient saliency and DRACH motif attribution
├── stitch_features.py            # Multi-panel figure assembly for PCA and t-SNE projections
├── stitch_confidence.py          # Multi-panel figure assembly for confidence KDE distributions
├── stitch_structure.py           # Multi-panel figure assembly for secondary-structure graphs
├── run_batch_train.sh            # Batch execution script for 5-fold CV and testing across 23 benchmarks
├── run_batch_interpretability.sh # Batch execution script for interpretability analyses
├── data/                         # Raw 41-nt FASTA files (training benchmarks and independent test sets)
├── data-processed/               # ViennaRNA-folded FASTA files with MFE secondary structures
├── train_model/                  # 5-fold cross-validation result CSVs (and saved weights) across 23 benchmarks
├── test_result/                  # Independent test set prediction scores and ensemble metrics across 23 benchmarks
├── result_inter/                 # Interpretability outputs (PCA/t-SNE, confidence KDE/CSVs, structure, saliency)
└── requirements.txt              # Python package dependencies
```

## 2. Environment Setup

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## 3. Data Preparation and Thermodynamic Folding

Raw 41-nt FASTA files are located under `data/train/benchmark/` and `data/test/independent/`. Pre-folded files with ViennaRNA minimum free energy (MFE) secondary structures are already provided in `data-processed/`. To re-run folding from scratch:

```bash
python process_data.py
```

## 4. Training and Independent Evaluation

To run stratified 5-fold cross-validation training and 5-fold ensemble independent testing on a specific benchmark (e.g., `Atlas_h_m1A`):

```bash
python run.py \
    --mode all \
    --model_type pinn \
    --train_data data-processed/train/benchmark/Atlas_h_m1A.fa \
    --test_data data-processed/test/independent/Atlas_h_m1A_Test.fa \
    --epochs 30 \
    --k_folds 5 \
    --lr 1e-3 \
    --batch_size 32
```

Supported `--model_type` options:
- `pinn`: Full **SPIN-RNA** contextual sequence backbone with physics-informed soft-bound regularization
- `gcn`: **RNA-GCN** spatial secondary-structure graph convolutional pathway
- `cnn`: **Motif-CNN** position-aware local motif-attention convolutional pathway

To run across all 23 benchmark datasets:

```bash
bash run_batch_train.sh
```

## 5. Model Interpretability Analyses

After training, run the four interpretability modules on held-out independent test sets (pre-computed results across all 23 benchmarks are provided in `result_inter/`):

```bash
# 1. Latent feature projections (PCA & t-SNE)
python visualize_features.py --dataset Atlas_h_m1A --model_type pinn --out_dir result_inter/features

# 2. Prediction confidence KDE distribution
python visualize_confidence.py --dataset h_b --model_type pinn --out_dir result_inter/confidence

# 3. Secondary-structure graph topology visualization
python visualize_structure.py --dataset Atlas_h_m6A --out_dir result_inter/structure

# 4. Nucleotide gradient saliency and motif attribution
python visualize_saliency.py --dataset h_b --model_type pinn --top_n 3 --out_dir result_inter/saliency
```
