import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

from dataset import load_real_data, RNADataset, Config

def plot_rna_graph(adj_matrix, seq_str, mfe, label, title, save_path, dataset_name, sample_idx):
    L = min(len(seq_str), adj_matrix.shape[0], Config.SEQ_LEN)
    adj = adj_matrix[:L, :L]
    
    G = nx.Graph()
    center_idx = L // 2
    
    # Define node colors based on Nucleotides
    color_map = {'A': '#2ca02c', 'U': '#1f77b4', 'C': '#9467bd', 'G': '#8c564b'}
    
    # 1. Add Nodes
    for i in range(L):
        G.add_node(i, base=seq_str[i])
        
    # 2. Add Backbone Edges (1D Sequence)
    backbone_edges = []
    for i in range(L - 1):
        backbone_edges.append((i, i+1))
        G.add_edge(i, i+1, type='backbone', weight=2.0)
        
    # 3. Add Spatial H-Bond Edges (from Adjacency Matrix, excluding i+1 backbone)
    hbond_edges = []
    for i in range(L):
        for j in range(i+2, L):
            if adj[i, j] > 0.5:
                # Add H-bond
                hbond_edges.append((i, j))
                G.add_edge(i, j, type='hbond', weight=1.0)
                
    # Layout using kamada_kawai which is great for minimizing energy-like graphs
    pos = nx.kamada_kawai_layout(G, scale=2.0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw Backbone
    nx.draw_networkx_edges(G, pos, edgelist=backbone_edges, width=2.5, edge_color='gray', ax=ax)
    
    # Draw H-Bonds
    nx.draw_networkx_edges(G, pos, edgelist=hbond_edges, width=1.5, edge_color='red', style='dashed', alpha=0.6, ax=ax)
    
    # Draw Nodes
    node_colors = []
    labels_dict = {}
    for i in range(L):
        if i == center_idx:
            node_colors.append('#ff7f0e') # Orange for center target
            labels_dict[i] = "A*" if seq_str[i] == 'A' else f"{seq_str[i]}*"
        else:
            base = seq_str[i]
            node_colors.append(color_map.get(base, '#333333'))
            labels_dict[i] = base
            
    nx.draw_networkx_nodes(G, pos, node_size=350, node_color=node_colors, edgecolors='black', linewidths=1.0, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=8, font_color='white', font_weight='bold', ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Legend for Elements
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Target Site (Center)', markerfacecolor='#ff7f0e', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='3D Spatial H-Bond (GCN Path)'),
        Line2D([0], [0], color='gray', lw=3, label='1D Sequence Backbone')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 0.70), fontsize=10, frameon=True)
    
    # Info Text Box
    info_text = (
        f"Dataset: {dataset_name}\n"
        f"Sample ID: #{sample_idx}\n"
        f"Sequence Topology\n"
        f"Length: {L} nt\n"
        f"Predicted: {'Positive' if label == 1 else 'Negative'}\n"
        f"Stem-Loop MFE: {mfe:.2f} kcal/mol\n"
        f"H-Bonds (Edges): {len(hbond_edges)}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray')
    ax.text(1.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
            
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if save_path.endswith('.png'):
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  -> Saved spatial graph to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="2D Structural Features for RNA_GCN")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset prefix")
    parser.add_argument('--out_dir', type=str, default='results_structure', help="Directory to save plots")
    parser.add_argument('--target_idx', type=int, default=-1, help="Index of sample to visualize (-1 for auto-select best)")
    args = parser.parse_args()

    test_data_path = f"data-processed/test/independent/{args.dataset}_Test.fa"
    if not os.path.exists(test_data_path):
        print(f"Error: {test_data_path} not found.")
        return

    # Structure plot represents purely the raw physical properties of the RNA Dataset itself.
    save_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    seqs, structs, mfes, labels = load_real_data(test_data_path)
    dataset = RNADataset(seqs, structs, mfes, labels)

    # Auto-select the BEST representative sample if target_idx < 0
    # Bio-rationale: The most "stable" structure is the positive sample with the lowest MFE
    if args.target_idx < 0:
        best_idx = 0
        min_mfe = float('inf')
        for i, (mfe_val, lbl) in enumerate(zip(mfes, labels)):
            if lbl == 1 and mfe_val < min_mfe:
                min_mfe = mfe_val
                best_idx = i
        args.target_idx = best_idx
        print(f"Auto-selected Sample #{best_idx} as it has the lowest MFE ({min_mfe:.2f}) among positive samples.")

    # Get sample with target index
    data_sample = dataset[args.target_idx]
    
    adj = data_sample['adj'].numpy()
    seq_str = seqs[args.target_idx]
    mfe = mfes[args.target_idx]
    label = labels[args.target_idx]
    
    title = f"SPIN-RNA Spatial Graph Convolution Mapping\nCase Study: RNA Secondary Structure Topology ({args.dataset} Sample #{args.target_idx})"
    print(f"Plotting Spatial Graph view on Sample #{args.target_idx} (True Label: {label})")
    plot_rna_graph(adj, seq_str, mfe, label, title, os.path.join(save_dir, f"contact_map_{args.target_idx}.png"), args.dataset, args.target_idx)

if __name__ == "__main__":
    main()
