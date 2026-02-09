import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import numpy as np
import logging
from config import * 
from func import * 
import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (Compose, LoadImaged, ToTensord, RandFlipd, RandRotated, 
                              RandZoomd, RandGaussianNoised, RandGaussianSmoothd, 
                              CastToTyped, ResizeD, NormalizeIntensityd, 
                              EnsureChannelFirstd)
from monai.data import CacheDataset
from datasets import RuntimeAugDataset
import torch.nn.functional as F
import copy
import gc

# =============================================================================
# MMD LOSS FUNCTIONS (Class-Conditional & Robust)
# =============================================================================

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
   
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2) 
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    
    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
    
    return sum(kernel_val)
def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Calcula a Loss MMD permitindo tamanhos de batch diferentes entre Source e Target.
    Isto é crucial para Conditional MMD.
    """
    n_source = int(source.size()[0])
    n_target = int(target.size()[0])
    
    kernels = gaussian_kernel(source, target, 
                              kernel_mul=kernel_mul, 
                              kernel_num=kernel_num, 
                              fix_sigma=fix_sigma)

    XX = kernels[:n_source, :n_source]   
    YY = kernels[n_source:, n_source:]  
    XY = kernels[:n_source, n_source:]
    YX = kernels[n_source:, :n_source]
    
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    
    return loss

def conditional_mmd_loss(source_features, source_labels, target_features, target_labels):
    """
    Calcula MMD separadamente para cada classe e faz a média.
    Garante que features de 'Nódulo' do Source se aproximam de 'Nódulo' do Target.
    """
    loss_mmd = torch.tensor(0.0, device=source_features.device)
    n_classes = 2
    count = 0

    for c in range(n_classes):
        # Máscaras booleanas para selecionar amostras da classe c
        source_mask = (source_labels == c).squeeze()
        target_mask = (target_labels == c).squeeze()

        # Verifica se existem amostras suficientes (>1) em ambos os domínios para calcular kernel
        if source_mask.sum() > 1 and target_mask.sum() > 1:
            s_feat = source_features[source_mask]
            t_feat = target_features[target_mask]
            
            loss_mmd += mmd_loss(s_feat, t_feat)
            count += 1

    return loss_mmd / count if count > 0 else loss_mmd

# =============================================================================
# TRAIN ONE EPOCH (With AMP & Concatenation)
# =============================================================================

def train_one_epoch(model, source_loader, target_loader, criterion, optimizer, scaler, device, alpha=0.25):
    model.train()
    total_loss, total_loss_sup, total_loss_mmd = 0, 0, 0
    total_samples = 0
    all_ids, all_labels, all_probs = [], [], []
    
    target_iter = iter(target_loader)

    for source_batch in source_loader:
        # --- 1. Preparação de Dados ---
        imgs_s = source_batch['image'].to(device, non_blocking=True)
        labels_s = source_batch['label'].to(device, non_blocking=True).float().unsqueeze(1)
        ids_s = source_batch['id']

        try:
            target_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_batch = next(target_iter)
        
        imgs_t = target_batch['image'].to(device, non_blocking=True)
        labels_t = target_batch['label'].to(device, non_blocking=True).float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        
        imgs_all = torch.cat([imgs_s, imgs_t], dim=0)
        labels_all = torch.cat([labels_s, labels_t], dim=0)

        with torch.cuda.amp.autocast():
            logits_all, features_all = model(imgs_all, return_features=True)
            loss_sup = criterion(logits_all, labels_all)

            batch_s_size = imgs_s.size(0)
            feat_s, feat_t = features_all[:batch_s_size], features_all[batch_s_size:]
            
            feat_s_norm = F.normalize(feat_s, p=2, dim=1)
            feat_t_norm = F.normalize(feat_t, p=2, dim=1)
            
            loss_mmd_val = conditional_mmd_loss(feat_s_norm, labels_s, feat_t_norm, labels_t)

            loss = loss_sup + (alpha * loss_mmd_val)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch_s_size
        total_loss_sup += loss_sup.item() * batch_s_size
        total_loss_mmd += loss_mmd_val.item() * batch_s_size
        total_samples += batch_s_size

        logits_s = logits_all[:batch_s_size]
        probs_s = torch.sigmoid(logits_s).detach().cpu().numpy().ravel()
        
        all_ids.extend(ids_s)
        all_labels.extend(labels_s.cpu().numpy().astype(int).ravel())
        all_probs.extend(probs_s)

    # --- Agregação por Paciente (Apenas Source) ---
    df = pd.DataFrame({'patient_id': all_ids, 'label': all_labels, 'probs': all_probs})
    patient_df = df.groupby('patient_id').agg({'label': lambda x: round(x.mean()), 'probs': 'mean'}).reset_index()
    patient_df = patient_df.dropna(subset=['probs', 'label'])

    threshold = 0.5
    patient_df['pred'] = (patient_df['probs'] > threshold).astype(int)

    y_true = patient_df['label'].to_numpy()
    y_pred = patient_df['pred'].to_numpy()

    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, average='macro') if len(np.unique(y_true)) > 1 else 0.0
    
    sens, spec = 0.0, 0.0
    if len(y_true) > 0 and len(np.unique(y_true)) > 1:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.size == 4:
            TN, FP, FN, TP = cm.ravel()
            sens = TP / (TP + FN + 1e-8)
            spec = TN / (TN + FP + 1e-8)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_loss_sup = total_loss_sup / total_samples if total_samples > 0 else 0
    avg_loss_mmd = total_loss_mmd / total_samples if total_samples > 0 else 0

    return avg_loss, avg_loss_sup, avg_loss_mmd, acc, sens, spec, f1

# =============================================================================
# VALIDATE FUNCTION
# =============================================================================

def validate(model, loader, criterion, device, test=False, best_threshold=0.5):
    model.eval()
    total_loss, total_samples = 0, 0
    all_ids, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True).float().unsqueeze(1)
            ids = batch['id']
            
            # Forward simples (não precisamos de features para validação normal)
            # Se o teu modelo retorna sempre tuple, usa [0]
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)
            
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            all_ids.extend(ids.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().astype(int).ravel())
            all_probs.extend(probs.detach().cpu().numpy().ravel())

    df = pd.DataFrame({'patient_id': all_ids, 'label': all_labels, 'probs': all_probs})
    patient_df = df.groupby('patient_id').agg({'label': lambda x: round(x.mean()), 'probs': 'mean'}).reset_index()
    patient_df = patient_df.dropna()

    if not test:
        try:
            fpr, tpr, thresholds = roc_curve(patient_df['label'], patient_df['probs'])
            best_threshold = thresholds[(tpr - fpr).argmax()]
        except:
            best_threshold = 0.5
    
    patient_df['pred'] = (patient_df['probs'] > best_threshold).astype(int)

    y_true = patient_df['label'].to_numpy()
    y_pred = patient_df['pred'].to_numpy()
    patient_probs = patient_df['probs'].to_numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    sens, spec = 0.0, 0.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        TN, FP, FN, TP = cm.ravel()
        sens = TP / (TP + FN + 1e-8)
        spec = TN / (TN + FP + 1e-8)
    
    try:
        auc = roc_auc_score(y_true, patient_probs)
    except:
        auc = 0.5

    return total_loss / total_samples, acc, sens, spec, f1, auc, best_threshold

# =============================================================================
# TRAIN KFOLD
# =============================================================================

def train_kfold(dataset, DATASET_MANY, ARCHITECTURE, criterion, optimizer, scheduler,
                num_epochs, batch_size, device, patience=15):

    global K_FOLDS, params, PRE_TRAINED, SAVE
    fold_results = []
    
    PERCENTAGE = 0.50 
    ALPHA = 0
    logging.info(f"Alpha: {ALPHA} | Supervised Mode: True | Class-Conditional MMD")

    items_list = [item for item in dataset.items]
    df = pd.DataFrame(items_list)
    df["image"] = df["image"].apply(lambda x: os.path.basename(x))
    
    folds_dict = create_split(df) 
    dataset_items = np.array(dataset.items)
    n_splits = 5
    
    log_dir = f"../Tensorboard/MMD_COND_SUP_P{PERCENTAGE}_{ALPHA}" 
    writer = SummaryWriter(log_dir=log_dir)

    static_base_transform = Compose([
        LoadImaged(keys=["image"], image_only=True), 
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        ResizeD(keys=["image"], spatial_size=(224, 224), mode="bilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        ToTensord(keys=["image", "label", "id", "dataset"]),
        CastToTyped(keys=['image'], dtype=np.float32), 
    ])

    runtime_aug = Compose([
        RandFlipd(keys=['image'], spatial_axis=[0], prob=0.4),
        RandRotated(keys=['image'], prob=0.4, range_x=[-0.25, 0.25], mode='bilinear', padding_mode="zeros"),
        RandZoomd(keys=['image'], prob=0.4, min_zoom=0.9, max_zoom=1.1, mode='area', padding_mode="minimum"),
        RandGaussianNoised(keys=['image'], std=0.005, prob=0.40),
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
    
    for DATASET in df['dataset'].unique():
        for FOLD in range(n_splits):
                
            model, optimizer, scheduler, criterion, params = training_config(ARCHITECTURE)
            model = model.to(device)
            
            # --- INICIALIZA O SCALER PARA AMP ---
            scaler = torch.cuda.amp.GradScaler()
            
            train_idx_final, val_idx_final, test_idx_final = create_set(df, PERCENTAGE, folds_dict, DATASET, FOLD)
            
            # Separação Source / Target
            df_train = df.iloc[train_idx_final].reset_index(drop=False) 
            mask_target = df_train['dataset'] == DATASET
            
            target_idx = df_train[mask_target]['index'].tolist()
            source_idx = df_train[~mask_target]['index'].tolist()

            df_val_temp = df.iloc[val_idx_final]
            val_idx_final = df_val_temp[df_val_temp['dataset'] == DATASET].index.tolist()

            print_overlay(df, train_idx_final, val_idx_final, test_idx_final)

            dataset_train_source = RuntimeAugDataset(full_dataset_cached, source_idx, transform=runtime_aug)
            dataset_train_target = RuntimeAugDataset(full_dataset_cached, target_idx, transform=runtime_aug)
            dataset_val = Subset(full_dataset_cached, val_idx_final)
            dataset_test = Subset(full_dataset_cached, test_idx_final)

            dl_kwargs = dict(
                batch_size=batch_size, 
                num_workers=3,          
                pin_memory=True,        
                persistent_workers=True,
                prefetch_factor=2       
            )

            # Samplers para balancear classes (importante para MMD condicional)
            source_labels = [dataset.items[idx]['label'] for idx in source_idx] 
            w_source = 1.0 / (np.bincount(source_labels) + 1e-6)
            sw_source = [w_source[label] for label in source_labels]
            source_sampler = WeightedRandomSampler(sw_source, len(sw_source), replacement=True)

            target_labels = [dataset.items[idx]['label'] for idx in target_idx] 
            if len(target_labels) > 0:
                w_target = 1.0 / (np.bincount(target_labels) + 1e-6)
                sw_target = [w_target[label] for label in target_labels]
                target_sampler = WeightedRandomSampler(sw_target, len(sw_target), replacement=True)
            else:
                target_sampler = None

            source_loader = DataLoader(dataset_train_source, sampler=source_sampler, **dl_kwargs)
            target_loader = DataLoader(dataset_train_target, sampler=target_sampler, **dl_kwargs)
            val_loader = DataLoader(dataset_val, shuffle=False, **dl_kwargs)
            test_loader = DataLoader(dataset_test, shuffle=False, **dl_kwargs)

            logging.info(f"Fold {FOLD} | Source Size: {len(source_idx)} | Target Size: {len(target_idx)}")
            
            best_criterium_value = -float("inf")
            counter = 0
            best_model_state = None
            best_train_metrics = None
            best_val_metrics = None
            optimal_threshold = 0.5
            ALPHA = 0

            for epoch in range(1, num_epochs + 1):

                ALPHA= ALPHA+0.01
                
                tr_loss, tr_sup, tr_mmd, tr_acc, tr_sens, tr_spec, tr_f1 = train_one_epoch(
                    model, source_loader, target_loader, criterion, optimizer, scaler, device, alpha=ALPHA
                )

                val_loss, val_acc, val_sens, val_spec, val_f1, val_auc, best_threshold = validate(
                    model, val_loader, criterion, device
                )

                if scheduler is not None:
                    scheduler.step(val_loss)

                logging.info(
                    f"Epoch {epoch:03d} | "
                    f"Tr Loss: {tr_loss:.4f} (Sup: {tr_sup:.4f} MMD: {tr_mmd:.4f}) | " 
                    f"Acc: {tr_acc:.4f} | Sens: {tr_sens:.4f} | Spec: {tr_spec:.4f} | F1: {tr_f1:.4f} || "
                    f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Sens: {val_sens:.4f} | Spec: {val_spec:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}"
                )
                
                print_tensorboard(writer, FOLD, DATASET, tr_loss, tr_sup, tr_mmd, tr_f1, val_loss, val_sens, val_spec, val_f1, val_auc, epoch)

                criterium_value = val_auc 

                if criterium_value > best_criterium_value + 1e-4:
                    best_criterium_value = criterium_value
                    counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_train_metrics = (tr_loss, tr_acc, tr_sens, tr_spec, tr_f1)
                    best_val_metrics = (val_loss, val_acc, val_sens, val_spec, val_f1, val_auc)
                    optimal_threshold = best_threshold
                else:
                    counter += 1

                if counter >= patience:
                    logging.info(f"Early stopping after {patience} epochs.")
                    break

            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            test_loss, test_acc, test_sens, test_spec, test_f1, test_auc, _ = validate(
                model=model, loader=test_loader, criterion=criterion, 
                device=device, test=True, best_threshold=optimal_threshold
            )
            
            test_metrics = (test_loss, test_acc, test_sens, test_spec, test_f1, test_auc)
            save_results_fold(fold_results, DATASET, FOLD, best_train_metrics, best_val_metrics, test_metrics)

            if SAVE:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(best_model_state, f"checkpoints/{DATASET}_fold{FOLD}_best.pth")

            print_results_fold(FOLD, fold_results)
            
            # Limpeza
            del model, optimizer, scaler
            torch.cuda.empty_cache()
            gc.collect()

    return fold_results