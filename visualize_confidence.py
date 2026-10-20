import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from dataset import load_real_data, RNADataset, Config
from model import get_model_class

def main():
    parser = argparse.ArgumentParser(description="Confidence and Error Analysis for SPIN-RNA")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset prefix")
    parser.add_argument('--model_type', type=str, default='pinn', help="Model architecture")
    parser.add_argument('--model_dir', type=str, default='train_model', help="Root dir of saved models")
    parser.add_argument('--out_dir', type=str, default='results_confidence', help="Directory to save plots and CSVs")
    args = parser.parse_args()

    test_data_path = f"data-processed/test/independent/{args.dataset}_Test.fa"
    if not os.path.exists(test_data_path):
        print(f"Error: {test_data_path} not found.")
        return

    model_weight_path = os.path.join(args.model_dir, args.model_type, args.dataset, "best_model.pth")
    if not os.path.exists(model_weight_path):
        alt_path = os.path.join(args.model_dir, args.model_type, f"{args.dataset}_all", "best_model.pth")
        if os.path.exists(alt_path):
            model_weight_path = alt_path
        else:
            print(f"Error: Weights not found at {model_weight_path}.")
            return

    save_dir = os.path.join(args.out_dir, args.model_type, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading test data for {args.dataset}...")
    seqs, structs, mfes, labels = load_real_data(test_data_path)
    dataset = RNADataset(seqs, structs, mfes, labels)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    ModelClass = get_model_class(args.model_type)
    model = ModelClass().to(Config.DEVICE)
    model.load_state_dict(torch.load(model_weight_path, map_location=Config.DEVICE))
    model.eval()

    print("Running inference across the test set...")
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for data_dict in loader:
            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    data_dict[k] = v.to(Config.DEVICE)
            
            if args.model_type == 'pinn':
                logits, _ = model(data_dict)
            else:
                logits, _ = model(data_dict)
                
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().reshape(-1))
            all_labels.extend(data_dict['label'].cpu().numpy().reshape(-1))
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)

    pred_types = []
    for p, l in zip(preds, all_labels):
        if p == 1 and l == 1:
            pred_types.append("True Positive")
        elif p == 0 and l == 0:
            pred_types.append("True Negative")
        elif p == 1 and l == 0:
            pred_types.append("False Positive")
        else:
            pred_types.append("False Negative")

    df = pd.DataFrame({
        "Sequence": seqs,
        "Secondary_Structure": structs,
        "MFE": mfes,
        "True_Label": all_labels,
        "Predicted_Prob": all_probs,
        "Predicted_Label": preds,
        "Type": pred_types
    })

    print("Generating Confidence Distribution Plot...")
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    sns.histplot(
        data=df, 
        x="Predicted_Prob", 
        hue="Type",
        hue_order=["True Positive", "True Negative", "False Positive", "False Negative"],
        palette=["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"], 
        bins=50, 
        kde=True,
        element="step"
    )
    
    plt.axvline(0.5, color='black', linestyle='--', linewidth=1.5, label='Decision Threshold (0.5)')
    display_name = "SPIN-RNA" if args.model_type == 'pinn' else args.model_type.upper()
    plt.title(f"Prediction Confidence KDE - {display_name} ({args.dataset})", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Model Confidence (Probability of Output)", fontsize=12)
    plt.ylabel("Number of Sequences", fontsize=12)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    acc = np.mean(df["True_Label"] == df["Predicted_Label"])
    plt.text(0.5, plt.ylim()[1]*0.9, f"Overall Accuracy: {acc:.2%}", ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    out_img = os.path.join(save_dir, "confidence_distribution.png")
    out_pdf = os.path.join(save_dir, "confidence_distribution.pdf")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()

    print("Extracting Error Analysis Dataframes...")
    ambiguous_df = df[(df["Predicted_Prob"] > 0.4) & (df["Predicted_Prob"] < 0.6)].copy()
    ambiguous_df["Ambiguity"] = 1 - abs(ambiguous_df["Predicted_Prob"] - 0.5) * 2
    ambiguous_df = ambiguous_df.sort_values(by="Ambiguity", ascending=False).head(20)
    
    err_df = df[((df["Type"] == "False Positive") & (df["Predicted_Prob"] > 0.8)) | 
                ((df["Type"] == "False Negative") & (df["Predicted_Prob"] < 0.2))].copy()
    err_df["Severity"] = abs(err_df["True_Label"] - err_df["Predicted_Prob"])
    err_df = err_df.sort_values(by="Severity", ascending=False).head(20)
    
    perf_df = df[df["Type"] == "True Positive"].sort_values(by="Predicted_Prob", ascending=False).head(20)

    ambiguous_df.to_csv(os.path.join(save_dir, "ambiguous_samples.csv"), index=False)
    err_df.to_csv(os.path.join(save_dir, "confident_errors.csv"), index=False)
    perf_df.to_csv(os.path.join(save_dir, "perfect_samples.csv"), index=False)
    
    print(f"Confidence Analysis plots and CSVs generated perfectly at: {save_dir}")

if __name__ == "__main__":
    main()
