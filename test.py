import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import csv
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, matthews_corrcoef, precision_score, average_precision_score

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
    
    try: roc_auc_val = roc_auc_score(y_true, y_probs)
    except: roc_auc_val = 0.5

    try: pr_auc_val = average_precision_score(y_true, y_probs)
    except: pr_auc_val = 0.0

    return {
        "ACC": acc, "Sn": sn, "Sp": sp, "Pre": pre, 
        "F1": f1, "MCC": mcc, "ROC_AUC": roc_auc_val, "PR_AUC": pr_auc_val
    }

def save_predictions_csv(targets, probs, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["True_Label", "Predicted_Score"])
        for t, p in zip(targets, probs): 
            writer.writerow([int(t), f"{p:.6f}"])
    print(f"[Test] Ensemble prediction scores saved to {filename}")

def save_test_metrics(metrics, result_file_path, model_type):
    save_dir = os.path.dirname(result_file_path)
    if not save_dir: save_dir = "."
    os.makedirs(save_dir, exist_ok=True)
    
    filename = os.path.join(save_dir, f"test_metrics_{model_type}_ensemble.csv")
    print(f"[Test] Saving ensemble metrics to {filename} ...")
    
    header = ["Model", "ACC", "Sn", "Sp", "Pre", "F1", "MCC", "ROC_AUC", "PR_AUC"]
    
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow([
            f"{model_type}_Ensemble", f"{metrics['ACC']:.4f}", f"{metrics['Sn']:.4f}",
            f"{metrics['Sp']:.4f}", f"{metrics['Pre']:.4f}", f"{metrics['F1']:.4f}",
            f"{metrics['MCC']:.4f}", f"{metrics['ROC_AUC']:.4f}", f"{metrics['PR_AUC']:.4f}"
        ])

def run_test_pipeline(data_file, model_path, result_file, model_type='pinn'):
    set_seed(Config.SEED)
    print(f"\n[Test] Device: {Config.DEVICE} | Model: {model_type} (5-Fold Ensemble Mode)")
    print(f"[Test] Data: {data_file}")
    
    model_dir = os.path.dirname(model_path)
    if not model_dir: model_dir = "."
    print(f"[Test] Searching for fold models in: {model_dir}")

    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found.")
        return

    seqs, structs, mfes, labels = load_real_data(data_file)
    test_loader = DataLoader(RNADataset(seqs, structs, mfes, labels), batch_size=Config.BATCH_SIZE, shuffle=False)
    
    ensemble_probs = np.zeros(len(labels))
    final_targets = np.array(labels) 
    models_found = 0

    ModelClass = get_model_class(model_type)
    
    for fold_idx in range(1, 6):
        fold_model_name = f"fold_{fold_idx}_model_{model_type}.pth"
        fold_model_path = os.path.join(model_dir, fold_model_name)
        
        if not os.path.exists(fold_model_path):
            print(f"[Warning] Model file not found: {fold_model_path}. Skipping.")
            continue
            
        print(f"--> Processing Fold {fold_idx}: {fold_model_name}")
        
        model = ModelClass().to(Config.DEVICE)
        model.load_state_dict(torch.load(fold_model_path, map_location=Config.DEVICE, weights_only=True))
        model.eval()
        
        fold_probs = []
        with torch.no_grad():
            for data_dict in test_loader:
                for k, v in data_dict.items():
                    if isinstance(v, torch.Tensor):
                        data_dict[k] = v.to(Config.DEVICE)
                
                if model_type == 'pinn':
                    logits, _ = model(data_dict)
                else:
                    logits, _ = model(data_dict)
                    
                probs = torch.sigmoid(logits).cpu().numpy()
                fold_probs.extend(probs)
        
        ensemble_probs += np.array(fold_probs).reshape(-1)
        models_found += 1

    if models_found == 0:
        print("[Error] No fold models found! Please run train.py first.")
        return

    avg_probs = ensemble_probs / models_found
    print(f"\n[Test] Ensemble completed using {models_found} models.")

    metrics = compute_all_metrics(final_targets, avg_probs)
    
    print("-" * 50)
    print(f"ENSEMBLE TEST SUMMARY ({model_type})")
    print("-" * 50)
    for k, v in metrics.items():
        print(f"{k:<8} : {v:.4f}")
    print("-" * 50)
    
    save_predictions_csv(final_targets, avg_probs, result_file)
    save_test_metrics(metrics, result_file, model_type)
