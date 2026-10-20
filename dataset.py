import torch
import numpy as np
import os
import random
import re
from torch.utils.data import Dataset

# ==========================================
# 1. 全局配置 (Global Configuration)
# ==========================================
class Config:
    SEED = 42
    SEQ_LEN = 41            # SPIN-RNA 序列长度 (基于先前的 benchmark 数据长度 41)
    VOCAB_SIZE = 4          # A, C, G, U
    BATCH_SIZE = 32
    EPOCHS = 30         
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# 2. 数据处理工具 (Sequence and Physics)
# ==========================================
NUCLEOTIDE_MAP = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'U': [0, 0, 0, 1],
    'T': [0, 0, 0, 1],  
}

MOTIF_RE = re.compile(r'[AG][AG]AC[ACU]')

def seq_to_onehot(seq: str) -> np.ndarray:
    L = len(seq)
    onehot = np.zeros((L, 4), dtype=np.float32)
    for i, ch in enumerate(seq):
        if ch in NUCLEOTIDE_MAP:
            onehot[i] = NUCLEOTIDE_MAP[ch]
    return onehot

def structure_to_adjacency(struct: str) -> np.ndarray:
    L = len(struct)
    adj = np.zeros((L, L), dtype=np.float32)

    # 相邻骨架键
    for i in range(L - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0

    # 配对碱基（氢键）
    stack = []
    for i, ch in enumerate(struct):
        if ch == '(':
            stack.append(i)
        elif ch == ')' and stack:
            j = stack.pop()
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    # 自连接
    np.fill_diagonal(adj, 1.0)
    return adj

def calc_physics_features(seq: str, mfe: float, motif_regex=MOTIF_RE) -> np.ndarray:
    gc_count = seq.count('G') + seq.count('C')
    gc_ratio = gc_count / max(len(seq), 1)

    motif_score = 0.0
    if len(seq) >= 26 and motif_regex.search(seq[15:26]):
        motif_score = 1.0

    mfe_norm = (mfe + 20.0) / 20.0
    mfe_norm = float(np.clip(mfe_norm, 0.0, 1.0))

    return np.array([mfe_norm, motif_score, gc_ratio], dtype=np.float32)

# ==========================================
# 3. 数据解析
# ==========================================
def load_real_data(file_path):
    sequences, structures, mfe_values, labels = [], [], [], []
    if not os.path.exists(file_path): raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue
            
        if line.startswith(">"):
            label = 1 if ">+" in line else 0
            i += 1
            
            if i >= len(lines): break
            seq = lines[i].upper()
            i += 1
            
            if i >= len(lines): break
            struct_line = lines[i]
            i += 1
            
            mfe_match = re.search(r'\(MFE:\s*([-.\d]+)\s*kcal/mol\)', struct_line)
            mfe = float(mfe_match.group(1)) if mfe_match else 0.0
            struct = re.sub(r'\s*\(MFE:.*\)', '', struct_line).strip()
            
            sequences.append(seq)
            structures.append(struct)
            mfe_values.append(mfe)
            labels.append(label)
        else:
            i += 1
    return sequences, structures, mfe_values, labels

# ==========================================
# 4. Dataset 设计 (多模态集成 Dataset)
# ==========================================
class RNADataset(Dataset):
    """
    通用 Dataset。由于 SPIN-RNA 的各个模型所需输入不同：
    PINN: (x_seq, x_phys, y)
    GCN: (x_seq, adj, y)
    CNN: (x_seq, y)
    Baseline/Transformer: 拼接形态的输入
    
    通过让 Dataset 直接返回一个打包好的字典或按照统一顺序返回值，让 Train 逻辑适配。
    为了无缝适配，返回一个字典，供模型端各取所需。
    """
    def __init__(self, sequences, structures, mfe_values, labels, max_len=None):
        self.sequences = sequences
        self.structures = structures
        self.mfe_values = mfe_values
        self.labels = labels
        self.max_len = max_len or Config.SEQ_LEN

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        struct = self.structures[idx]
        mfe = self.mfe_values[idx]
        label = self.labels[idx]

        x_seq_np = seq_to_onehot(seq)  # (L, 4)
        L = x_seq_np.shape[0]
        
        # padding
        pad_len = max(0, self.max_len - L)
        
        # Seq
        x_seq = np.pad(x_seq_np, ((0, pad_len), (0, 0)), 'constant')
        x_seq = torch.tensor(x_seq, dtype=torch.float32)
        
        # Adj
        adj_np = structure_to_adjacency(struct) # (L, L)
        adj = np.pad(adj_np, ((0, pad_len), (0, pad_len)), 'constant')
        adj = torch.tensor(adj, dtype=torch.float32)
        
        # Phys
        x_phys_np = calc_physics_features(seq, mfe)
        x_phys = torch.tensor(x_phys_np, dtype=torch.float32)
        
        # Label
        y = torch.tensor([label], dtype=torch.float32)
        
        return {
            'x_seq': x_seq,
            'adj': adj,
            'x_phys': x_phys,
            'label': y
        }
