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

from adabn import Norm, configure_model 
from datasets import *

# ==============================================================================
# FUNÇÃO DE TESTE COM ADABN (Atualizada para retornar vetores)
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

            # Garante que IDs são lista de strings/ints
            if isinstance(ids, torch.Tensor):
                all_ids.extend(ids.cpu().numpy())
            else:
                all_ids.extend(ids) 

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

    # --- RETORNO ATUALIZADO: Inclui vetores raw ---
    return avg_loss, acc, sens, spec, f1, auc, best_threshold, patient_df['patient_id'].tolist(), y_true, patient_df['probs'].to_numpy()


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

    logging.info("Carregando CacheDataset...")
    full_dataset_cached = CacheDataset(data=dataset_items, transform=post_spacing_transform, cache_rate=1.0)
    
    fold_results = []

    for DATASET_NAME in df['dataset'].unique():
        
        # === Listas para Acumulação Global (Pooled) ===
        global_ids = []
        global_y_true = []
        global_y_probs = []
        # ==============================================

        for FOLD in range(n_splits):
            
            model, _, _, criterion, _ = training_config(ARCHITECTURE)
            model = model.to(device)
            
            train_idx, val_idx, train_idx_final, val_idx_final, test_idx_final = create_set(df, folds_dict, DATASET_NAME, FOLD)
            
            dataset_test = RuntimeAugDataset(full_dataset_cached, test_idx_final, transform=val_transform)

            dl_kwargs = dict(
                batch_size=len(dataset_test), 
                num_workers=3,          
                pin_memory=True,        
                persistent_workers=True,
                prefetch_factor=2       
            )
            
            test_loader = DataLoader(dataset_test, shuffle=False, **dl_kwargs)

            logging.info(f"📊Test Dataset Size: {len(dataset_test)}")

            path = f"checkpoints_dn_0/{DATASET_NAME}_fold{FOLD}.pth"
            if not os.path.exists(path):
                logging.warning(f"Checkpoint não encontrado: {path}. Pulando Fold.")
                continue
                
            best_model_state_from_disk = torch.load(path, map_location=device)
            model.load_state_dict(best_model_state_from_disk)
            
            # Chama a função e recebe os vetores extras
            test_loss, test_acc, test_sens, test_spec, test_f1, test_auc, best_threshold, p_ids, p_true, p_probs = test_with_adabn(
                model=model, loader=test_loader, criterion=criterion, device=device, TTA_MOMENTUM=1, TTA_RESET_STATS=True
            )
   
            logging.info(
                f"Fold {FOLD} | Test (AdaBN) - AUC: {test_auc:.4f} | F1: {test_f1:.4f}"
            )
            
            # === ACUMULAÇÃO DOS DADOS DO FOLD ===
            global_ids.extend(p_ids)
            global_y_true.extend(p_true)
            global_y_probs.extend(p_probs)
            # ====================================

            test_metrics = (test_loss, test_acc, test_sens, test_spec, test_f1, test_auc)
            save_results_fold(fold_results, DATASET_NAME, FOLD, None, None, test_metrics)
            
        
        # === CÁLCULO DA POOLED AUC (APÓS TODOS OS FOLDS) ===
        logging.info(f"\n{'*'*60}")
        logging.info(f" RELATÓRIO POOLED (AdaBN) - DATASET: {DATASET_NAME}")
        logging.info(f"{'*'*60}")
        
        global_y_true = np.array(global_y_true)
        global_y_probs = np.array(global_y_probs)
        
        if len(np.unique(global_y_true)) > 1:
            # Cálculo da AUC Global
            pooled_auc = roc_auc_score(global_y_true, global_y_probs)
            
            # Cálculo do Threshold Ótimo Global (Oracle) apenas para reportar métricas ideais
            fpr, tpr, thresholds = roc_curve(global_y_true, global_y_probs)
            best_thresh = thresholds[(tpr - fpr).argmax()]
            
            pooled_preds = (global_y_probs > best_thresh).astype(int)
            pooled_acc = accuracy_score(global_y_true, pooled_preds)
            pooled_f1 = f1_score(global_y_true, pooled_preds, average='macro')
            
            cm = confusion_matrix(global_y_true, pooled_preds, labels=[0, 1])
            
            logging.info(f"Pooled AUC: {pooled_auc:.4f}")
            logging.info(f"Pooled F1 (Best Thresh {best_thresh:.3f}): {pooled_f1:.4f}")
            logging.info(f"Pooled Acc: {pooled_acc:.4f}")
            logging.info(f"Pooled Confusion Matrix:\n{cm}")
            
            # Guardar CSV consolidado
            df_pooled = pd.DataFrame({
                "patient_id": global_ids,
                "label": global_y_true,
                "prob_adabn": global_y_probs
            })
            df_pooled.to_csv(f"pooled_adabn_{DATASET_NAME}.csv", index=False)
            logging.info(f"Resultados pooled guardados em: pooled_adabn_{DATASET_NAME}.csv")
        else:
            logging.warning("Não foi possível calcular Pooled AUC (apenas 1 classe presente nos dados acumulados).")

    return fold_results