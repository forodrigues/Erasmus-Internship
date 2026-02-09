import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import numpy as np
import logging
from config import *
from func import *
import os
import pandas as pd
from monai.data import CacheDataset, set_track_meta
from monai.transforms import (Compose, LoadImaged, ToTensord, RandFlipd, RandRotated, 
                              RandZoomd, RandGaussianNoised, RandGaussianSmoothd, 
                              CastToTyped, ResizeD, NormalizeIntensityd, 
                              EnsureChannelFirstd)

# Importa a classe Norm e funções auxiliares do seu arquivo adabn.py
from adabn import Norm, configure_model 

# ==============================================================================
# FUNÇÃO DE TESTE COM ADABN (Test-Time Adaptation)
# ==============================================================================
def test_with_adabn(model, loader, criterion, device, TTA_MOMENTUM=1.0, TTA_RESET_STATS=True):
   
    adabn_model = Norm(
        model=model, 
        eps=1e-5, 
        momentum=TTA_MOMENTUM, 
        reset_stats=TTA_RESET_STATS, 
        no_stats=False
    )
    
    adabn_model = adabn_model.to(device)
    
    total_loss, total_samples = 0, 0
    all_ids, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True).float().unsqueeze(1)
            ids = batch['id']

            outputs = adabn_model(imgs)
            
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)
            
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            all_ids.extend(ids.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().astype(int).ravel())
            all_probs.extend(probs.detach().cpu().numpy().ravel())

    df = pd.DataFrame({'patient_id': all_ids, 'label': all_labels, 'probs': all_probs})   
    patient_df = df.groupby('patient_id').agg({'label': lambda x: round(x.mean()), 'probs': 'mean'}).reset_index()

    if len(patient_df['label'].unique()) > 1:
        fpr, tpr, thresholds = roc_curve(patient_df['label'], patient_df['probs'])
        best_threshold = thresholds[(tpr - fpr).argmax()]
        auc = roc_auc_score(patient_df['label'], patient_df['probs'])
    else:
        best_threshold = 0.5
        auc = 0.5

    patient_df['pred'] = (patient_df['probs'] > best_threshold).astype(int)

    y_true = patient_df['label'].to_numpy()
    y_pred = patient_df['pred'].to_numpy()
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0,0,0,0)
    sens = TP / (TP + FN + 1e-8)
    spec = TN / (TN + FP + 1e-8)
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    return avg_loss, acc, sens, spec, f1, auc, best_threshold


def train_kfold(dataset, DATASET_MANY, ARCHITECTURE, criterion, optimizer, scheduler,
                num_epochs, batch_size, device, patience=PATIENCE):

    global K_FOLDS, params, PRE_TRAINED, SAVE
    fold_results = []
    
    items_list = [item for item in dataset.items]
    df = pd.DataFrame(items_list)
    df["image"] = df["image"].apply(lambda x: os.path.basename(x))
    n_splits = 5

    folds_dict = create_split(df)
    dataset_items = np.array(dataset.items)

    static_base_transform = Compose([
        LoadImaged(keys=["image"], image_only=True), 
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        ResizeD(keys=["image"], spatial_size=(224, 224), mode="bilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        ToTensord(keys=["image", "label", "id", "dataset"]),
        CastToTyped(keys=['image'], dtype=np.float32), 
    ])
    
    logging.info("Iniciando carregamento do CacheDataset (RAM)...")
    full_dataset_cached = CacheDataset(
        data=dataset_items, 
        transform=static_base_transform, 
        cache_rate=1.0,  
        num_workers=4    
    )
    logging.info(f"Dataset carregado. Tamanho: {len(full_dataset_cached)}")

    for DATASET_NAME in df['dataset'].unique():
        for FOLD in range(n_splits):
            
            model, _, _, criterion, _ = training_config(ARCHITECTURE)
            model = model.to(device)
            
            train_idx, val_idx, train_idx_final, val_idx_final, test_idx_final = create_set(df, folds_dict, DATASET_NAME, FOLD)
            
            dataset_test = Subset(full_dataset_cached, test_idx_final)

            dl_kwargs = dict(
                batch_size=len(dataset_test), 
                num_workers=3,          
                pin_memory=True,        
                persistent_workers=True,
                prefetch_factor=2       
            )

            
            test_loader = DataLoader(dataset_test, shuffle=False, **dl_kwargs)
            logging.info(f"📊Test Dataset Size: {len(dataset_test)}")

            path = f"checkpointsalpha1dn/{DATASET_NAME}_fold{FOLD}_best.pth"
            if not os.path.exists(path):
                logging.warning(f"Checkpoint não encontrado: {path}. Pulando Fold.")
                continue
                
            best_model_state_from_disk = torch.load(path, map_location=device)
            model.load_state_dict(best_model_state_from_disk)
            
            
            test_loss, test_acc, test_sens, test_spec, test_f1, test_auc, best_threshold = test_with_adabn(model=model, loader=test_loader, criterion=criterion, device=device,TTA_MOMENTUM=1, TTA_RESET_STATS=True)
   
            logging.info(
                f"Test (AdaBN) - Loss: {test_loss:.4f} | "
                f"Acc: {test_acc:.4f} | "
                f"Sens: {test_sens:.4f} | "
                f"Spec: {test_spec:.4f} | "
                f"F1: {test_f1:.4f} | "
                f"AUC: {test_auc:.4f}"
            )
            
            test_metrics = (test_loss, test_acc, test_sens, test_spec, test_f1, test_auc)
            save_results_fold(fold_results, DATASET_NAME, FOLD, None, None, test_metrics)
            

            
    return fold_results