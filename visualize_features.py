import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from dataset import load_real_data, RNADataset, Config
from model import get_model_class

def plot_scatter(features, labels, title, save_path):
    plt.figure(figsize=(8, 6))
    
    label_names = ["Negative (0)", "Positive (1)"]
    str_labels = [label_names[int(lbl)] for lbl in labels]
    
    sns.scatterplot(
        x=features[:, 0], 
        y=features[:, 1],
        hue=str_labels,
        palette=["#1f77b4", "#d62728"], 
        s=40,
        alpha=0.8,
        edgecolor="w",
        linewidth=0.5
    )
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.legend(title="Class", loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Save as PDF as well
    if save_path.endswith('.png'):
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        
    plt.close()
    print(f"  -> Saved {title} plot to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Feature Visualization (t-SNE & PCA) for SPIN-RNA")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset prefix, e.g., 'Atlas_h_m1A'")
    parser.add_argument('--model_type', type=str, default='pinn', help="Model architecture: 'pinn', 'gcn', 'cnn'")
    parser.add_argument('--model_dir', type=str, default='train_model', help="Root dir of saved models")
    parser.add_argument('--out_dir', type=str, default='visualizations', help="Directory to save plots")
    parser.add_argument('--max_samples', type=int, default=2000, help="Max samples to plot to avoid clutter")
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
            print(f"Error: Model weights not found in {model_weight_path} or {alt_path}")
            return

    save_dir = os.path.join(args.out_dir, args.model_type, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    print(f"=== Interpreting {args.model_type} on {args.dataset} ===")
    
    seqs, structs, mfes, labels = load_real_data(test_data_path)
    dataset = RNADataset(seqs, structs, mfes, labels)
    
    if len(dataset) > args.max_samples:
        indices = np.random.choice(len(dataset), args.max_samples, replace=False)
        dataset = Subset(dataset, indices)
    
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    ModelClass = get_model_class(args.model_type)
    model = ModelClass().to(Config.DEVICE)
    model.load_state_dict(torch.load(model_weight_path, map_location=Config.DEVICE))
    model.eval()

    features_list = []
    labels_list = []
    
    def hook_fn(module, input):
        features_list.append(input[0].detach().cpu().numpy())

    # Attach intercept hooks differently according to structure
    if args.model_type == 'pinn':
        hook_handle = model.classifier_head.register_forward_pre_hook(hook_fn)
    elif args.model_type == 'gcn':
        hook_handle = model.classifier.register_forward_pre_hook(hook_fn)
    elif args.model_type == 'cnn':
        hook_handle = model.classifier.register_forward_pre_hook(hook_fn)
    else:
        print("Unknown model_type for hooking.")
        return

    print("Extracting features...")
    with torch.no_grad():
        for data_dict in loader:
            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    data_dict[k] = v.to(Config.DEVICE)
            
            # Forward pass to trigger hook
            if args.model_type == 'pinn':
                _ = model(data_dict) 
            else:
                _ = model(data_dict)
                
            labels_list.extend(data_dict['label'].cpu().numpy().reshape(-1))
    
    hook_handle.remove()

    X = np.concatenate(features_list, axis=0)
    Y = np.array(labels_list)
    print(f"Extracted feature shape: {X.shape}")

    display_name = "SPIN-RNA" if args.model_type == 'pinn' else args.model_type.upper()
    print("Running PCA...")
    pca = PCA(n_components=2, random_state=Config.SEED)
    X_pca = pca.fit_transform(X)
    plot_scatter(X_pca, Y, f"PCA of {display_name} ({args.dataset})", os.path.join(save_dir, "pca_plot.png"))

    print("Running t-SNE (this might take a minute)...")
    try:
        n_samples = X.shape[0]
        if n_samples < 5:
            print(f"  -> Skipped t-SNE: Not enough samples ({n_samples}, needs at least 5).")
        else:
            dyn_perplexity = int(min(30, max(2, (n_samples - 1) // 3)))
            tsne = TSNE(n_components=2, perplexity=dyn_perplexity, random_state=Config.SEED, init='random')
            X_tsne = tsne.fit_transform(X)
            plot_scatter(X_tsne, Y, f"t-SNE of {display_name} ({args.dataset})", os.path.join(save_dir, "tsne_plot.png"))
    except Exception as e:
        print(f"  -> t-SNE Failed with error: {e}")

    print(f"Interpretability analysis finished! Plots saved to: {save_dir}")

if __name__ == "__main__":
    main()
