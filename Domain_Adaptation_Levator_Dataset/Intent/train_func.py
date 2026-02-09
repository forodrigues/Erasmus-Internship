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
from monai.transforms import (
    Compose, LoadImaged, ResizeD, ToTensord, RandFlipd, RandRotated,
    RandGaussianNoised, RandGaussianSmoothd, CastToTyped, EnsureTyped,
    RandAdjustContrastd, RandShiftIntensityd, EnsureChannelFirstd, NormalizeIntensityd,
    CropForegroundd, KeepLargestConnectedComponentd, SpatialPadd, Lambdad, Spacingd
)
from adabn import Tent, configure_model, collect_params, copy_model_and_optimizer 
from datasets import *

# ==============================================================================
# FUNÇÃO DE TESTE COM TENT (Test-Time Adaptation)
# ==============================================================================
def test_with_tent(model, loader, criterion, device, TENT_STEPS, TENT_LR, episodic=True):
    
    # Prepara o modelo para TENT (congela pesos, ativa BN para treino)
    model = configure_model(model)
    params, _ = collect_params(model)
    optimizer = optim.Adam(params, lr=TENT_LR) 
    
    tent_model = Tent(model, optimizer, steps=TENT_STEPS, episodic=episodic)
    tent_model = tent_model.to(device)
    
    total_loss, total_samples = 0, 0
    all_ids, all_labels, all_probs = [], [], []

    for batch in loader:
        imgs = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True).float().unsqueeze(1)
        ids = batch['id'] # Lista de strings

        outputs = tent_model(imgs)
        
        with torch.no_grad():
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)
            
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            # --- CORREÇÃO DO ERRO 'list' object has no attribute 'cpu' ---
            if isinstance(ids, torch.Tensor):
                all_ids.extend(ids.cpu().numpy().tolist())
            else:
                all_ids.extend(list(ids)) # IDs de pacientes costumam ser strings
            # -------------------------------------------------------------

            all_labels.extend(labels.cpu().numpy().astype(int).ravel())
            all_probs.extend(probs.detach().cpu().numpy().ravel())
        

    # Agregação por Paciente
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

    # Retorna métricas e vetores para o cálculo da Pooled AUC
    return avg_loss, acc, sens, spec, f1, auc, best_threshold, patient_df['patient_id'].tolist(), y_true, patient_df['probs'].to_numpy()


def train_kfold(dataset, DATASET_MANY, ARCHITECTURE, criterion, optimizer, scheduler,
                num_epochs, batch_size, device, patience=PATIENCE):

    global K_FOLDS, params, PRE_TRAINED, SAVE
    fold_results = []
    
    TENT_STEPS = 10 
    TENT_LR = 1e-2 

    items_list = [item for item in dataset.items]
    df = pd.DataFrame(items_list)
    df["image"] = df["image"].apply(lambda x: os.path.basename(x))
    n_splits = 5

    folds_dict = create_split(df)
    dataset_items = np.array(dataset.items)

    post_spacing_transform = Compose([
        LoadImaged(keys=["image"]), 
        EnsureChannelFirstd(keys=["image"]),
        Lambdad(keys="image", func=aspect_ratio_pres), 
        ResizeD(keys=["image"], spatial_size=(224, 224), mode="bilinear"),
        ToTensord(keys=["image", "label", "subfolder", "id", "image_name"]),
        CastToTyped(keys=['image'], dtype=np.float32),
        EnsureTyped(keys=['image'])
    ])

    val_transform = Compose([
        NormalizeIntensityd(keys=["image"], nonzero=False),
        CastToTyped(keys=['image'], dtype=np.float32),
        EnsureTyped(keys=['image'])
    ])

    logging.info("Iniciando carregamento do CacheDataset (RAM)...")
    full_dataset_cached = CacheDataset(data=dataset_items, transform=post_spacing_transform, cache_rate=1.0)

    for DATASET_NAME in df['dataset'].unique():
        
        # --- VARIÁVEIS PARA POOLED AUC ---
        global_ids, global_y_true, global_y_probs = [], [], []

        for FOLD in range(n_splits):
            logging.info(f"\n{'='*20} FOLD {FOLD} (TENT Adaptation) {'='*20}")
            
            model, _, _, criterion, _ = training_config(ARCHITECTURE)
            model = model.to(device)
            
            _, _, _, _, test_idx_final = create_set(df, folds_dict, DATASET_NAME, FOLD)
            dataset_test = RuntimeAugDataset(full_dataset_cached, test_idx_final, transform=val_transform)

            test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=True, num_workers=3, pin_memory=True)
            
            path = f"checkpoints_dn_0/{DATASET_NAME}_fold{FOLD}.pth"
            if not os.path.exists(path):
                logging.warning(f"Checkpoint não encontrado: {path}. Pulando Fold.")
                continue
                
            model.load_state_dict(torch.load(path, map_location=device))

            # Executa TENT
            res = test_with_tent(model, test_loader, criterion, device, TENT_STEPS, TENT_LR, episodic=False)
            t_loss, t_acc, t_sens, t_spec, t_f1, t_auc, t_thresh, p_ids, p_true, p_probs = res

            logging.info(f"Test (TENT) Fold {FOLD} - AUC: {t_auc:.4f} | F1: {t_f1:.4f}")
            
            # Acumulação Global
            global_ids.extend(p_ids)
            global_y_true.extend(p_true)
            global_y_probs.extend(p_probs)

            test_metrics = (t_loss, t_acc, t_sens, t_spec, t_f1, t_auc)
            save_results_fold(fold_results, DATASET_NAME, FOLD, None, None, test_metrics)
            
        # --- RELATÓRIO POOLED GLOBAL ---
        logging.info(f"\n{'*'*60}\n RELATÓRIO POOLED FINAL (TENT) - DATASET: {DATASET_NAME}\n{'*'*60}")
        g_true = np.array(global_y_true)
        g_probs = np.array(global_y_probs)

        if len(np.unique(g_true)) > 1:
            pooled_auc = roc_auc_score(g_true, g_probs)
            # Threshold Oracle (Best for whole dataset)
            fpr, tpr, thresholds = roc_curve(g_true, g_probs)
            best_t = thresholds[(tpr - fpr).argmax()]
            
            logging.info(f"Pooled AUC: {pooled_auc:.4f} | Oracle Threshold: {best_t:.4f}")
            
            df_pooled = pd.DataFrame({'id': global_ids, 'true': g_true, 'prob': g_probs})
            df_pooled.to_csv(f"pooled_results_TENT_{DATASET_NAME}.csv", index=False)
            
    return fold_results