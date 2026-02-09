import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score,cohen_kappa_score,roc_curve,accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from collections import Counter
import numpy as np
import logging
from config import *
from func import *
import os
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
import matplotlib.pyplot as plt
from datasets import Dataset2D,RuntimeAugDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (
    Compose, LoadImaged, ResizeD, ToTensord, RandFlipd, RandRotated,
    RandGaussianNoised, RandGaussianSmoothd, CastToTyped, EnsureTyped,
    RandAdjustContrastd, RandShiftIntensityd,EnsureChannelFirstd,NormalizeIntensityd,
    CropForegroundd, KeepLargestConnectedComponentd,SpatialPadd,Lambdad,Spacingd
)
from monai.data import CacheDataset, set_track_meta
import copy


# --- FUNÇÃO 1: TREINO POR ÉPOCA ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_samples = 0, 0
    all_labels, all_probs = [], []
    #save_images(loader, save_folder="Saved_images_train", max_images=180)
    for batch in loader:
        imgs = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        probs = torch.sigmoid(outputs)
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        all_labels.extend(labels.cpu().numpy().astype(int).ravel())
        all_probs.extend(probs.detach().cpu().numpy().ravel())

    # Métricas rápidas de treino (Slice level)
    probs_np = np.array(all_probs)
    labels_np = np.array(all_labels)
    preds_np = (probs_np > 0.5).astype(int)
    
    acc = accuracy_score(labels_np, preds_np)
    f1 = f1_score(labels_np, preds_np, average='macro')
    
    cm = confusion_matrix(labels_np, preds_np, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)

    return total_loss / total_samples, acc, sens, spec, f1

# --- FUNÇÃO 2: VALIDAÇÃO (RETORNA BEST THRESHOLD) ---
def validate(model, loader, criterion, device, test=False, best_threshold=0.5):
    model.eval()
    total_loss, total_samples = 0, 0
    all_ids, all_labels, all_probs = [], [], []
    #save_images(loader, save_folder="Saved_images_val_adela", max_images=180)
    
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True).float().unsqueeze(1)
            ids = batch['id']
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)
            
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            all_ids.extend(ids)
            all_labels.extend(labels.cpu().numpy().astype(int).ravel())
            all_probs.extend(probs.detach().cpu().numpy().ravel())

    # Agregação por Paciente (CRUCIAL)
    df = pd.DataFrame({'patient_id': all_ids, 'label': all_labels, 'probs': all_probs})
    patient_df = df.groupby('patient_id').agg({
        'label': lambda x: round(x.mean()), 
        'probs': 'mean'
    }).reset_index()

    y_true = patient_df['label'].to_numpy()
    patient_probs = patient_df['probs'].to_numpy()

    # Calcular AUC
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, patient_probs)
        
        # --- CÁLCULO DO THRESHOLD ÓTIMO (SE FOR VALIDAÇÃO) ---
        if not test:
            fpr, tpr, thresholds = roc_curve(y_true, patient_probs)
            # Youden's J statistic
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
        else:
            optimal_threshold = best_threshold # No teste, usa o que foi passado (média das validações)
    else:
        auc = 0.5
        optimal_threshold = 0.5

    # Aplicar Threshold
    y_pred = (patient_probs > optimal_threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)

    return total_loss / total_samples, acc, sens, spec, f1, auc, optimal_threshold

# --- FUNÇÃO 3: OBTER RESULTADOS FINAIS DO FOLD ---
def get_patient_results(model, loader, device):
    model.eval()
    all_ids, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True).float().unsqueeze(1)
            ids = batch['id']

            outputs = model(imgs)
            probs = torch.sigmoid(outputs)

            all_ids.extend(ids)
            all_labels.extend(labels.cpu().numpy().astype(int).ravel())
            all_probs.extend(probs.cpu().numpy().ravel())

    df = pd.DataFrame({'patient_id': all_ids, 'label': all_labels, 'probs': all_probs})
    patient_df = df.groupby('patient_id').agg({
        'label': lambda x: round(x.mean()), 
        'probs': 'mean'
    }).reset_index()

    return (
        patient_df['patient_id'].tolist(),
        patient_df['label'].to_numpy(),
        patient_df['probs'].to_numpy()
    )


def train_kfold(dataset, DATASET_MANY, ARCHITECTURE, criterion, optimizer, scheduler,
                num_epochs, batch_size, device,patience):

    global K_FOLDS, params, PRE_TRAINED, SAVE, PATIENCE
    
    fold_results = []
    
    PERCENTAGE = 0
    
    logging.info(f"Percentage: {PERCENTAGE} | Strategy: Pooled Cross-Validation (Honest Thresholding)")

    items_list = [item for item in dataset.items]
    df = pd.DataFrame(items_list)
    df["image"] = df["image"].apply(lambda x: os.path.basename(x))
    
    log_dir = f"../Tensorboard/Pooled_{PERCENTAGE}" 
    writer = SummaryWriter(log_dir=log_dir)

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

    train_transform = Compose([
        RandFlipd(['image'], spatial_axis=[0], prob=0.3),
        RandRotated(keys=['image'], prob=0.3, range_x=[-0.2618, 0.2618], mode='bilinear', padding_mode="zeros"),
        RandGaussianNoised(keys=['image'], std=0.005, prob=0.3),
        RandShiftIntensityd(keys=['image'], offsets=(-0.2, 0.2), prob=0.3),
        RandAdjustContrastd(keys=['image'], gamma=(0.6, 1.4), prob=0.3),
        NormalizeIntensityd(keys=["image"], nonzero=False),
        CastToTyped(keys=['image'], dtype=np.float32),
        EnsureTyped(keys=['image'])
    ])

    val_transform = Compose([
        NormalizeIntensityd(keys=["image"], nonzero=False),
        CastToTyped(keys=['image'], dtype=np.float32),
        EnsureTyped(keys=['image'])
    ])
    # ====================================
    
    logging.info("Iniciando carregamento do CacheDataset (RAM)...")
    full_dataset_cached = CacheDataset(
        data=dataset_items, 
        transform=post_spacing_transform, 
        cache_rate=1.0 
    )
    logging.info(f"Dataset carregado. Tamanho: {len(full_dataset_cached)}")

    for DATASET in df['dataset'].unique():
        
        # Variáveis Globais (Pooled)
        global_ids = []
        global_y_true = []
        global_y_probs = []
        fold_optimal_thresholds = [] # Guarda o melhor thresh de validação de cada fold

        for FOLD in range(K_FOLDS): 
            logging.info(f"\n{'='*40} FOLD {FOLD}/{K_FOLDS-1} {'='*40}")
            
            model, optimizer, scheduler, criterion, params = training_config(ARCHITECTURE)
            train_idx_final, val_idx_final, test_idx_final = create_set(df, PERCENTAGE, folds_dict, DATASET, FOLD)
            print_overlay(df, train_idx_final, val_idx_final, test_idx_final)
            

            
            dataset_train = RuntimeAugDataset(full_dataset_cached, train_idx_final, transform=train_transform)
            dataset_val = RuntimeAugDataset(full_dataset_cached, val_idx_final, transform=val_transform)
            dataset_test = RuntimeAugDataset(full_dataset_cached, test_idx_final, transform=val_transform)

            dl_kwargs = dict(batch_size=batch_size, num_workers=3, pin_memory=True, persistent_workers=True, prefetch_factor=2)
            train_loader = DataLoader(dataset_train, shuffle=True, **dl_kwargs)
            val_loader = DataLoader(dataset_val, shuffle=False, **dl_kwargs)
            test_loader = DataLoader(dataset_test, shuffle=False, **dl_kwargs)

            print_loader_distribution(DATASET, FOLD, train_loader, val_loader, test_loader)

            best_criterium_value = -float("inf")
            counter = 0
            best_model_state = None
            best_train_metrics = None
            best_val_metrics = None
            
            current_fold_optimal_thresh = 0.5 

            train_labels = df.iloc[train_idx_final]['label'].values
            num_neg = (train_labels == 0).sum()
            num_pos = (train_labels == 1).sum()
            pos_weight = torch.tensor([num_neg / max(num_pos, 1e-5)], device=device).float()
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            for epoch in range(1, num_epochs + 1):
                train_loss, train_acc, train_sens, train_spec, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
                
                val_loss, val_acc, val_sens, val_spec, val_f1, val_auc, epoch_best_thresh = validate(model, val_loader, criterion, device, test=False)


                _, _,_, _, _, test_auc,_= validate(model, test_loader, criterion, device, test=False)

                if scheduler is not None: scheduler.step(val_loss)

               
                logging.info(f"Epoch {epoch:03d} | Tr_Loss: {train_loss:.4f} | Tr_F1: {train_f1:.4f} | Val_AUC: {val_auc:.4f} | Test_AUC: {test_auc:.4f}")
                
                print_tensorboard(writer, FOLD, DATASET, train_loss, train_f1, val_loss, val_sens, val_spec, val_f1, val_auc, epoch)

                if val_auc > best_criterium_value + 1e-4 and epoch >2:
                    best_criterium_value = val_auc
                    counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_train_metrics = (train_loss, train_acc, train_sens, train_spec, train_f1)
                    best_val_metrics = (val_loss, val_acc, val_sens, val_spec, val_f1, val_auc)
                    
                    # Atualizamos o threshold ótimo com o desta época vencedora
                    current_fold_optimal_thresh = epoch_best_thresh
                else:
                    counter += 1

                if counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break

            # --- FIM DO FOLD ---
            model.load_state_dict(best_model_state)
            
            # Adiciona o threshold aprendido à lista para cálculo da média global
            fold_optimal_thresholds.append(current_fold_optimal_thresh)
            
            test_loss, test_acc, test_sens, test_spec, test_f1, test_auc, _ = validate(
                model, 
                test_loader, 
                criterion, 
                device, 
                test=True, 
                best_threshold=current_fold_optimal_thresh 
            )
            
            test_metrics = (test_loss, test_acc, test_sens, test_spec, test_f1, test_auc)
            save_results_fold(fold_results, DATASET, FOLD, best_train_metrics, best_val_metrics, test_metrics)
            print_results_fold(FOLD, fold_results)

            if SAVE:
                os.makedirs("checkpoints_dn_0", exist_ok=True)
                path = f"checkpoints_dn_0/{DATASET}_fold{FOLD}.pth"
                torch.save(best_model_state, path)

            # 2. Acumulação Global
            p_ids, p_true, p_probs = get_patient_results(model, test_loader, device)
            global_ids.extend(p_ids)
            global_y_true.extend(p_true)
            global_y_probs.extend(p_probs)
            
            logging.info(f"Fold {FOLD} acumulado. Threshold Validação: {current_fold_optimal_thresh:.4f}")

            del model, optimizer, train_loader, val_loader, test_loader
            torch.cuda.empty_cache()

        # === RESULTADOS GLOBAIS ===
        logging.info(f"\n{'*'*60}")
        logging.info(f" RELATÓRIO POOLED (GLOBAL) - DATASET: {DATASET}")
        logging.info(f"{'*'*60}")

        global_y_true = np.array(global_y_true)
        global_y_probs = np.array(global_y_probs)

        if len(np.unique(global_y_true)) > 1:
            global_auc = roc_auc_score(global_y_true, global_y_probs)
            
            # --- Threshold Honesto (Average Validation Threshold) ---
            # Este é o que deves usar para Accuracy, F1, etc.
            avg_val_thresh = np.mean(fold_optimal_thresholds)
            
            # --- Threshold Oracle (Para comparação apenas) ---
            fpr, tpr, thresholds = roc_curve(global_y_true, global_y_probs)
            best_oracle_thresh = thresholds[(tpr - fpr).argmax()]
            
            global_preds = (global_y_probs > avg_val_thresh).astype(int)
            
            global_acc = accuracy_score(global_y_true, global_preds)
            global_f1 = f1_score(global_y_true, global_preds, average='macro')
            
            cm = confusion_matrix(global_y_true, global_preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            global_sens = tp / (tp + fn + 1e-8)
            global_spec = tn / (tn + fp + 1e-8)

            logging.info(f"Global AUC:         {global_auc:.4f}")
            logging.info(f"Global Accuracy:    {global_acc:.4f}")
            logging.info(f"Global F1-Score:    {global_f1:.4f}")
            logging.info(f"Global Sensitivity: {global_sens:.4f}")
            logging.info(f"Global Specificity: {global_spec:.4f}")
            logging.info(f"Threshold Usado (Avg Val): {avg_val_thresh:.4f}")
            logging.info(f"Best Threshold: {best_oracle_thresh:.4f}")
            logging.info(f"Global Confusion Matrix:\n{cm}")
            
            # Guardar CSV
            df_global = pd.DataFrame({
                'patient_id': global_ids,
                'true_label': global_y_true,
                'prob_score': global_y_probs,
                'pred_label': global_preds
            })
            df_global.to_csv(f"pooled_results_{DATASET}.csv", index=False)
            
        else:
            logging.error("Erro Global: Apenas 1 classe presente.")

        
    return fold_results