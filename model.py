import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================================================================
# 1. PINN 物理信息架构
# ====================================================================
class PhysicsInformedHead(nn.Module):
    def __init__(self, phys_in_dim, hidden_dim=32):
        super(PhysicsInformedHead, self).__init__()
        self.phys_proj = nn.Sequential(
            nn.Linear(phys_in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

    def forward(self, p_methyl, x_phys):
        thermo_bound = self.phys_proj(x_phys)
        physics_residual = torch.relu(p_methyl - thermo_bound)
        return torch.mean(physics_residual ** 2)

class Ultimate_Fusion_PINN(nn.Module):
    def __init__(self, seq_in_dim=4, phys_dim=3, hidden_dim=128):
        super(Ultimate_Fusion_PINN, self).__init__()
        self.conv1 = nn.Conv1d(seq_in_dim, hidden_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim//2, bidirectional=True, batch_first=True)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim, 4, batch_first=True), num_layers=1)
        self.classifier_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.physics_head = PhysicsInformedHead(phys_in_dim=phys_dim)
        
    def forward(self, data_dict):
        x_seq = data_dict['x_seq']
        x_phys = data_dict['x_phys']
        
        x = x_seq.transpose(1, 2)
        x = F.relu(self.conv1(x)).transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.transformer(x)
        logits = self.classifier_head(x.mean(dim=1))
        
        # 兼容 Train loss 的物理约束
        phys_loss = self.physics_head(torch.sigmoid(logits), x_phys)
        return logits, phys_loss

# ====================================================================
# 2. 图卷积神经网络 (RNA_GCN)
# ====================================================================
class RNA_GCN(nn.Module):
    def __init__(self, in_features=4, hidden_dim=64):
        super(RNA_GCN, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, data_dict):
        x_seq = data_dict['x_seq']
        adj_matrix = data_dict['adj']
        
        hidden = F.relu(self.fc1(x_seq))   
        gcn_out = torch.bmm(adj_matrix, hidden)
        gcn_out = F.relu(self.fc2(gcn_out))
        graph_embed = gcn_out.mean(dim=1)  
        
        logits = self.classifier(graph_embed)
        return logits, gcn_out

# ====================================================================
# 3. 基序注意力网络 (Motif_Attention_CNN)
# ====================================================================
class Motif_Attention_CNN(nn.Module):
    def __init__(self, seq_in_dim=4, hidden_dim=64):
        super(Motif_Attention_CNN, self).__init__()
        self.conv1 = nn.Conv1d(seq_in_dim, hidden_dim, kernel_size=5, padding=2)
        
        self.attention_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data_dict):
        x_seq = data_dict['x_seq']
        
        x = x_seq.transpose(1, 2)  
        motif_features = F.relu(self.conv1(x)) 
        motif_features = motif_features.transpose(1, 2) 
        
        attn_logits = self.attention_score(motif_features) 
        attn_weights = F.softmax(attn_logits, dim=1) 
        context_vector = torch.sum(attn_weights * motif_features, dim=1) 
        
        logits = self.classifier(context_vector)
        return logits, attn_weights

# ====================================================================
# 4. 获取模型路由 (Get Model Class)
# ====================================================================
def get_model_class(model_type: str):
    """
    统一的模型获取接口
    """
    model_map = {
        'pinn': Ultimate_Fusion_PINN,
        'gcn': RNA_GCN,
        'cnn': Motif_Attention_CNN,
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model_type: {model_type}. Options are: {list(model_map.keys())}")
    
    return model_map[model_type]
