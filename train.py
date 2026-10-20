import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, matthews_corrcoef, precision_score
import numpy as np
import os
import csv

from dataset import Config, set_seed, RNADataset, load_real_data
from model import get_model_class

def compute_all_metrics(y_true, y_probs):
    y_pred = np.round(y_probs)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = tp / (tp + fn + 1e-10)
    sp = tn / (tn + fp + 1e-10)
    pre = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    try: auc_val = roc_auc_score(y_true, y_probs)
    except: auc_val = 0.5
    return {"ACC": acc, "Sn": sn, "Sp": sp, "Pre": pre, "F1": f1, "MCC": mcc, "AUC": auc_val}

def train_one_epoch(model, loader, criterion, optimizer, model_type):
    model.train()
    total_loss = 0
    for data_dict in loader:
        # Move all dict values to device
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.to(Config.DEVICE)
        
        optimizer.zero_grad()
        
        # Determine specific loss computation based on model type
        if model_type == 'pinn':
            logits, phys_loss = model(data_dict)
            loss = criterion(logits, data_dict['label']) + 0.1 * phys_loss
        else:
            logits, _ = model(data_dict)
            loss = criterion(logits, data_dict['label'])
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, model_type):
    model.eval()
    probs, targets = [], []
    total_loss = 0
    with torch.no_grad():
        for data_dict in loader:
            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    data_dict[k] = v.to(Config.DEVICE)
            
            if model_type == 'pinn':
                logits, phys_loss = model(data_dict)
                loss = criterion(logits, data_dict['label']) + 0.1 * phys_loss
            else:
                logits, _ = model(data_dict)
                loss = criterion(logits, data_dict['label'])
                
            total_loss += loss.item()
            probs.extend(torch.sigmoid(logits).cpu().numpy())
            targets.extend(data_dict['label'].cpu().numpy())
            
    return np.array(probs), np.array(targets), total_loss / len(loader)

def save_training_logs(fold_results, save_dir, model_type):
    filename = os.path.join(save_dir, f"training_log_{model_type}.csv")
    metrics = ["ACC", "Sn", "Sp", "Pre", "F1", "MCC", "AUC"]
    
    print(f"\n[Log] Saving training metrics to {filename} ...")
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["Fold"] + metrics
        writer.writerow(header)
        
        num_folds = len(fold_results["ACC"])
        for i in range(num_folds):
            row = [f"Fold_{i+1}"]
            for m in metrics:
                row.append(f"{fold_results[m][i]:.4f}")
            writer.writerow(row)
            
        writer.writerow([])
        avg_row = ["Average"]
        std_row = ["Std Dev"]
        for m in metrics:
            avg_val = np.mean(fold_results[m])
            std_val = np.std(fold_results[m])
            avg_row.append(f"{avg_val:.4f}")
            std_row.append(f"{std_val:.4f}")
            
        writer.writerow(avg_row)
        writer.writerow(std_row)

def run_training_pipeline(data_file, save_path, model_type='pinn', k_folds=5, epochs=30):
    set_seed(Config.SEED)
    print(f"\n[Train] Device: {Config.DEVICE} | Model: {model_type} | K-Fold: {k_folds} | Epochs: {epochs}")
    print(f"[Train] Data: {data_file}")
    
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        return

    seqs, structs, mfes, labels = load_real_data(data_file)
    dataset = RNADataset(seqs, structs, mfes, labels)
    labels_np = np.array(labels)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=Config.SEED)

    ModelClass = get_model_class(model_type)
    
    fold_results = {"ACC": [], "Sn": [], "Sp": [], "Pre": [], "F1": [], "MCC": [], "AUC": []}
    global_best_acc = 0.0

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels_np)), labels_np)):
        print(f"\n{'='*15} Fold {fold_idx + 1}/{k_folds} {'='*15}")
            
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=Config.BATCH_SIZE, shuffle=False)
        
        model = ModelClass().to(Config.DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        fold_best_acc = 0.0
        fold_best_metrics = {}
        
        for epoch in range(epochs):
            t_loss = train_one_epoch(model, train_loader, criterion, optimizer, model_type)
            v_probs, v_targets, v_loss = validate(model, val_loader, criterion, model_type)
            metrics = compute_all_metrics(v_targets, v_probs)
            print(f"[Fold {fold_idx+1}][Epoch {epoch+1}] T_Loss: {t_loss:.4f}, V_Loss: {v_loss:.4f}, ACC: {metrics['ACC']:.4f}, Sn: {metrics['Sn']:.4f}, Sp: {metrics['Sp']:.4f}, Pre: {metrics['Pre']:.4f}, AUC: {metrics['AUC']:.4f}", flush=True)
            scheduler.step()            
            
            if metrics["ACC"] > fold_best_acc:
                fold_best_acc = metrics["ACC"]
                fold_best_metrics = metrics
                
                fold_model_name = f"fold_{fold_idx+1}_model_{model_type}.pth"
                torch.save(model.state_dict(), os.path.join(save_dir, fold_model_name))
                
                if fold_best_acc > global_best_acc:
                    global_best_acc = fold_best_acc
                    torch.save(model.state_dict(), save_path)
        
        print(f"Fold {fold_idx+1} Finished. Best ACC: {fold_best_metrics['ACC']:.4f}")
        for k, v in fold_best_metrics.items(): 
            fold_results[k].append(v)

    save_training_logs(fold_results, save_dir, model_type)

    print(f"\n{'#'*20} Training Summary {'#'*20}")
    print(f"Global Best Accuracy: {global_best_acc:.4f}")
    print(f"Best model saved to : {save_path}")
    print("-" * 50)
    print(f"{'Metric':<10} | {'Average':<10} | {'Std Dev':<10}")
    print("-" * 50)
    metric_order = ["ACC", "Sn", "Sp", "Pre", "F1", "MCC", "AUC"]
    for m in metric_order:
        print(f"{m:<10} | {np.mean(fold_results[m]):.4f}     | +/- {np.std(fold_results[m]):.4f}")
    print("-" * 50)
