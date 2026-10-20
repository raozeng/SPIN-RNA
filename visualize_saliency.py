import os
import argparse
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
torch.backends.cudnn.enabled = False

from dataset import load_real_data, RNADataset, Config
from model import get_model_class

def find_drach_motifs(seq_str):
    pattern = r'[AGU][AG]AC[ACU]'
    matches = []
    for match in re.finditer(pattern, seq_str):
        matches.append((match.start(), match.end()))
    return matches

def compute_saliency(model, data_dict, model_type):
    model.eval()
    
    x_seq = data_dict['x_seq'].to(Config.DEVICE)
    x_seq.requires_grad_(True)
    data_dict['x_seq'] = x_seq
    
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor) and k != 'x_seq':
            data_dict[k] = v.to(Config.DEVICE)
    
    # 按照模型类型前向传播
    if model_type == 'pinn':
        logits, _ = model(data_dict)
    else:
        logits, _ = model(data_dict)
        
    prob = torch.sigmoid(logits)
    model.zero_grad()
    prob.backward()
    
    # Saliency is absolute sum along one-hot encode dimension
    saliency = x_seq.grad.data.abs().sum(dim=-1).squeeze().cpu().numpy()
    return prob.item(), saliency

def plot_single_sequence(ax, seq_str, saliency, sample_idx, confidence, drach_motifs):
    sal_min, sal_max = np.min(saliency), np.max(saliency)
    if sal_max > sal_min:
        saliency_norm = (saliency - sal_min) / (sal_max - sal_min)
    else:
        saliency_norm = np.zeros_like(saliency)
        
    cmap = plt.cm.Reds # Using reds for SPIN-RNA
    L = len(seq_str)
    
    for i in range(min(L, len(saliency_norm))):
        color = cmap(saliency_norm[i])
        rect = patches.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='white', linewidth=1)
        ax.add_patch(rect)
        text_color = "white" if saliency_norm[i] > 0.6 else "black"
        ax.text(i + 0.5, 0.5, seq_str[i], color=text_color, ha='center', va='center', fontsize=7, zorder=3)
        
    for start, end in drach_motifs:
        m_len = end - start
        rect = patches.Rectangle((start, -0.05), m_len, 1.1, edgecolor='#1f77b4', facecolor='none', linewidth=2.0, zorder=5)
        ax.add_patch(rect)
        ax.text(start + m_len/2, -0.4, "DRACH", color='#1f77b4', ha='center', va='center', fontsize=8, fontweight='bold')
        
    # 添加中心位点 A*
    center_idx = L // 2
    if seq_str[center_idx] == 'A':
        ax.axvline(x=center_idx + 0.5, color='red', linestyle='--', linewidth=1.5, zorder=4)
        ax.text(center_idx + 0.5, 1.35, "A*", color='red', ha='center', va='center', fontsize=10, fontweight='bold')
        
    ax.set_xlim(-0.5, min(L, Config.SEQ_LEN) + 0.5)
    ax.set_ylim(-1.0, 1.8)
    ax.set_xticks(np.arange(0, min(L, Config.SEQ_LEN), 5) + 0.5)
    ax.set_xticklabels(np.arange(1, min(L, Config.SEQ_LEN)+1, 5), fontsize=8)
    ax.set_yticks([])
    
    disp_seq = seq_str[:12] + "..." if len(seq_str) > 12 else seq_str
    
    ax.set_title(
        f"Sample #{sample_idx} | Confidence: {confidence:.4f} | DRACH motifs: {len(drach_motifs)} | Seq: {disp_seq}", 
        fontsize=11, fontweight='bold', pad=15
    )
    
    # 绘制迷你 Colorbar
    cax = ax.inset_axes([0.4, -0.4, 0.2, 0.08])
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    cb.set_label("Normalized Saliency Score", size=8)
    cb.ax.tick_params(labelsize=8)
    
def main():
    parser = argparse.ArgumentParser(description="RNA Sequence Saliency Map / Attribution")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset prefix")
    parser.add_argument('--model_type', type=str, default='pinn', help="Model architecture")
    parser.add_argument('--model_dir', type=str, default='train_model', help="Root dir of saved models")
    parser.add_argument('--out_dir', type=str, default='results_saliency', help="Directory to save plots")
    parser.add_argument('--top_n', type=int, default=3, help="Number of top confident samples to plot")
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

    print("Loading data and model...")
    seqs, structs, mfes, labels = load_real_data(test_data_path)
    dataset = RNADataset(seqs, structs, mfes, labels)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    ModelClass = get_model_class(args.model_type)
    model = ModelClass().to(Config.DEVICE)
    model.load_state_dict(torch.load(model_weight_path, map_location=Config.DEVICE))
    model.eval()
    
    print("Scanning instances for highest confidence...")
    all_probs = []
    with torch.no_grad():
        for data_dict in loader:
            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    data_dict[k] = v.to(Config.DEVICE)
                    
            if args.model_type == 'pinn':
                logits, _ = model(data_dict)
            else:
                logits, _ = model(data_dict)
                
            all_probs.extend(torch.sigmoid(logits).cpu().numpy().reshape(-1))
    
    all_probs = np.array(all_probs)
    labels_np = np.array(labels)
    
    pos_indices = np.where(labels_np == 1)[0]
    
    # Filter highly saturated prob regions (>0.99) to prevent gradient vanishing on visualizations
    valid_pos = pos_indices[all_probs[pos_indices] < 0.99]
    if len(valid_pos) < args.top_n:
        valid_pos = pos_indices  # fallback if distribution is completely hyper-confident
        
    sorted_pos = valid_pos[np.argsort(all_probs[valid_pos])[::-1]]
    top_indices = sorted_pos[:args.top_n]
    
    print(f"Found top {args.top_n} candidates. Generating saliency maps...")
    
    fig, axes = plt.subplots(args.top_n, 1, figsize=(12, 2.5 * args.top_n))
    if args.top_n == 1:
        axes = [axes]
        
    for ax_idx, idx in enumerate(top_indices):
        data_dict = {
            'x_seq': dataset[idx]['x_seq'].unsqueeze(0),
            'adj': dataset[idx]['adj'].unsqueeze(0),
            'x_phys': dataset[idx]['x_phys'].unsqueeze(0),
            'label': dataset[idx]['label'].unsqueeze(0)
        }
        
        seq_str = seqs[idx]
        conf, saliency = compute_saliency(model, data_dict, args.model_type)
        motifs = find_drach_motifs(seq_str)
        plot_single_sequence(axes[ax_idx], seq_str, saliency, idx, conf, motifs)

    plt.tight_layout(pad=3.0)
    out_png = os.path.join(save_dir, "saliency_heatmaps.png")
    out_pdf = os.path.join(save_dir, "saliency_heatmaps.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f"Plot saved to {out_png} and {out_pdf}")

if __name__ == "__main__":
    main()
