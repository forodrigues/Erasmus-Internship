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
from datasets import Dataset2D, RuntimeAugDataset
import pandas as pd
from itertools import zip_longest
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (Compose, LoadImaged, ToTensord, RandFlipd, RandRotated, 
                              RandZoomd, RandGaussianNoised, RandGaussianSmoothd, 
                              CastToTyped, EnsureTyped, ResizeD, NormalizeIntensityd, 
                              EnsureChannelFirstd)
from monai.data import CacheDataset, set_track_meta
import gc
import copy
import torch.nn.functional as F

# Desliga rastreio de metadados para economizar RAM e CPU
set_track_meta(False)



def calculate_im_score(logits):
    """
    Calcula o Information Maximization (IM) Score.
    IM = H(mean(probs)) - mean(H(probs))
    Quanto maior, melhor (indica diversidade global e certeza local).
    """

    probs = torch.sigmoid(logits)
    probs_2class = torch.stack([1.0 - probs, probs], dim=1).squeeze()
    entropy_per_sample = -torch.sum(probs_2class * torch.log(probs_2class + 1e-6), dim=1)
    mean_conditional_entropy = torch.mean(entropy_per_sample)
    mean_probs = torch.mean(probs_2class, dim=0)
    marginal_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-6))
    im_score = marginal_entropy - mean_conditional_entropy

    return im_score.item()

def calculate_snd(features, temperature=0.05):
    
    if features.size(0) <= 1:
        return 0.0
    
    features = F.normalize(features, p=2, dim=1)
    sim_mat = torch.mm(features, features.t())
    sim_mat = sim_mat / temperature
    mask = torch.eye(sim_mat.size(0), device=sim_mat.device).bool()
    sim_mat.masked_fill_(mask, -float('inf'))
    p_ij = F.softmax(sim_mat, dim=1)
    entropy = -torch.sum(p_ij * torch.log(p_ij + 1e-8), dim=1)
    avg_entropy = entropy.mean()
    num_neighbors = features.size(0) - 1
    normalized_entropy = avg_entropy / np.log(num_neighbors)
    snd_value = np.exp(-normalized_entropy.item())
    
    return snd_value
# ==============================================================================
# FUNÇÃO DE LOSS (CORAL - Correlation Alignment)
# ==============================================================================

def coral_loss(source, target):
    d = source.size(1) 
    ns, nt = source.size(0), target.size(0)

    xm = source - torch.mean(source, 0, keepdim=True)
    xc = xm.t() @ xm / (ns - 1)

    xmt = target - torch.mean(target, 0, keepdim=True)
    xct = xmt.t() @ xmt / (nt - 1)

    loss = torch.sum((xc - xct) ** 2)
    loss = loss / (4 * d * d)
    return loss


# ==============================================================================
# FUNÇÃO DE TREINO COM AMP (Mixed Precision)
# ==============================================================================
def train_one_epoch(model, sup_loader, unsup_loader, criterion, optimizer, scaler, DATASET, device, alpha):

    model.train()
    # Inicializar todas as variáveis de acumulação
    total_loss, total_loss_sup, total_loss_coral = 0, 0, 0
    total_samples = 0
    
    all_ids, all_labels, all_probs = [], [], []
    
    unsup_iter = iter(unsup_loader) if alpha > 0 else None

    for sup_batch in sup_loader:
        
        imgs_sup = sup_batch['image'].to(device, non_blocking=True)
        labels = sup_batch['label'].to(device, non_blocking=True).float().unsqueeze(1)
        ids = sup_batch['id']
        
        if len(imgs_sup) < 2:
            continue

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(): # Garante que o bloco está identado no autocast
            outputs_sup, features_sup = model(imgs_sup, return_features=True)
            loss_sup = criterion(outputs_sup, labels)

            loss_coral = torch.tensor(0.0, device=device)
            loss = loss_sup 
            
            if alpha > 0:
                try:
                    unsup_batch = next(unsup_iter)
                except StopIteration:
                    unsup_iter = iter(unsup_loader)
                    unsup_batch = next(unsup_iter)
                
                imgs_unsup = unsup_batch['image'].to(device, non_blocking=True)
                
                # Só calcula coral se tivermos amostras suficientes
                if imgs_unsup.size(0) > 1:
                    _, features_unsup = model(imgs_unsup, return_features=True)
                    
                    # --- CORREÇÃO 1: FLATTEN ---
                    # Garante que as features são [Batch, 768] antes de ir para o kernel
                    if len(features_sup.shape) > 2:
                        features_sup = torch.flatten(features_sup, 1)
                    if len(features_unsup.shape) > 2:
                        features_unsup = torch.flatten(features_unsup, 1)
                    min_bs = min(features_sup.size(0), features_unsup.size(0))
                    
                    # Corta ambos para terem o mesmo tamanho (min_bs)
                    features_sup = features_sup[:min_bs]
                    features_unsup = features_unsup[:min_bs]

                    
                    loss_coral_val = coral_loss(features_sup, features_unsup)
                    
                    loss_coral = torch.clamp(loss_coral_val, min=0.0)
                    
                    loss = loss_sup + alpha * loss_coral 
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        probs = torch.sigmoid(outputs_sup)
        bs_sup = labels.size(0)
        
        total_loss += loss.item() * bs_sup
        total_loss_sup += loss_sup.item() * bs_sup
        total_loss_coral += loss_coral.item() * bs_sup 
        total_samples += bs_sup

        all_ids.extend(ids.numpy() if not torch.is_tensor(ids) else ids.cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy().astype(int).ravel())
        all_probs.extend(probs.detach().cpu().numpy().ravel())

    df = pd.DataFrame({'patient_id': all_ids, 'label': all_labels, 'probs': all_probs})
    patient_df = df.groupby('patient_id').agg({'label': lambda x: round(x.mean()), 'probs': 'mean'}).reset_index()
    patient_df['pred'] = (patient_df['probs'] > 0.5).astype(int)
    
    y_true = patient_df['label'].to_numpy()
    y_pred = patient_df['pred'].to_numpy()
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0,0,0,0)
    sens = TP / (TP + FN + 1e-8)
    spec = TN / (TN + FP + 1e-8)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_loss_sup = total_loss_sup / total_samples if total_samples > 0 else 0.0
    avg_loss_coral = total_loss_coral / total_samples if total_samples > 0 else 0.0

    return avg_loss, avg_loss_sup, avg_loss_coral, acc, sens, spec, f1

# FUNÇÃO DE VALIDAÇÃO
# ==============================================================================

def validate(model, loader, criterion, DATASET, device, test=False, best_threshold=None, unsup_loader=None):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_ids, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True).float().unsqueeze(1)
            ids = batch['id']
            
            cls_preds = model(imgs, return_features=False) 
            
            loss = criterion(cls_preds, labels)
            probs = torch.sigmoid(cls_preds)
            
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            all_ids.extend(ids.numpy() if not torch.is_tensor(ids) else ids.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().astype(int).ravel())
            all_probs.extend(probs.cpu().numpy().ravel())

    # --- Patient-Level Aggregation ---
    df = pd.DataFrame({'patient_id': all_ids, 'label': all_labels, 'probs': all_probs})
    patient_df = df.groupby('patient_id').agg({'label': lambda x: round(x.mean()), 'probs': 'mean'}).reset_index()

    y_true = patient_df['label'].to_numpy()
    y_probs = patient_df['probs'].to_numpy()

    if best_threshold is None:
        if len(np.unique(y_true)) > 1:
            fpr, tpr, thresholds = roc_curve(y_true, y_probs)
            best_threshold = thresholds[(tpr - fpr).argmax()]
        else:
            best_threshold = 0.5
    
    y_pred = (y_probs > best_threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0,0,0,0)
    sens = TP / (TP + FN + 1e-8)
    spec = TN / (TN + FP + 1e-8)
    
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = 0.5

    avg_loss = total_loss / total_samples if total_samples > 0 else 0

    snd_score = 0.0
    im_score = 0.0
    
    if unsup_loader is not None:
        target_features = []
        target_logits = []
        
        with torch.no_grad():
            for batch in unsup_loader:
                imgs = batch['image'].to(device, non_blocking=True)
                
                # AQUI MANTÉM-SE IGUAL (porque return_features=True devolve 2 coisas)
                logits, features = model(imgs, return_features=True)
                
                target_features.append(features.cpu())
                target_logits.append(logits.cpu())
        
        if len(target_features) > 0:
            all_features = torch.cat(target_features, dim=0)
            snd_score = calculate_snd(all_features)
            
            all_logits = torch.cat(target_logits, dim=0)
            im_score = calculate_im_score(all_logits)

    if test:
        return avg_loss, acc, sens, spec, f1, auc, best_threshold
    else:
        return avg_loss, acc, sens, spec, f1, auc, best_threshold, snd_score, im_score

# ==============================================================================
# FUNÇÃO PRINCIPAL (OTIMIZADA PARA CORAL)
# ==============================================================================
def train_kfold(dataset, DATASET_MANY, ARCHITECTURE, criterion, optimizer_cls, scheduler_cls,
                num_epochs, batch_size, device, patience=PATIENCE):

    global K_FOLDS, params, PRE_TRAINED, SAVE
    fold_results = []

    items_list = [item for item in dataset.items]
    df = pd.DataFrame(items_list)
    df["image"] = df["image"].apply(lambda x: os.path.basename(x))
    n_splits = 5
    
    ALPHA = 100
    
    logging.info(f"Alpha: {ALPHA}")
    log_dir = f"../Tensorboard/CORAL_ALPHA{ALPHA}" 

    writer = SummaryWriter(log_dir=log_dir)

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
        cache_rate=1,  
        num_workers=4    
    )
    logging.info(f"Dataset carregado. Tamanho: {len(full_dataset_cached)}")
    
    for DATASET_NAME in df['dataset'].unique():
        for FOLD in range(n_splits):
            
            model, optimizer, scheduler, criterion, params = training_config(ARCHITECTURE)
            model = model.to(device)
            
            scaler = torch.cuda.amp.GradScaler()

            train_idx, val_idx, train_idx_final, val_idx_final, test_idx_final = create_set(df, folds_dict, DATASET_NAME, FOLD)
            print_overlay(df, train_idx_final, val_idx_final, test_idx_final)

            dataset_train = RuntimeAugDataset(full_dataset_cached, train_idx_final, transform=runtime_aug)
            dataset_unsup = RuntimeAugDataset(full_dataset_cached, train_idx , transform=runtime_aug)
            dataset_val = Subset(full_dataset_cached, val_idx_final)
            dataset_val_unsup = Subset(full_dataset_cached, val_idx)

            dataset_test = Subset(full_dataset_cached, test_idx_final)

            dl_kwargs = dict(
                batch_size=batch_size, 
                num_workers=3,          
                pin_memory=True,        
                persistent_workers=True,
                prefetch_factor=2       
            )

            unsup_loader = DataLoader(dataset_unsup, shuffle=True, **dl_kwargs)
            train_loader = DataLoader(dataset_train, shuffle=True, **dl_kwargs)
            val_loader = DataLoader(dataset_val, shuffle=False, **dl_kwargs)
            val_unsup_loader = DataLoader(dataset_val_unsup, shuffle=False, **dl_kwargs)

            test_loader = DataLoader(dataset_test, shuffle=False, **dl_kwargs)

            print_loader_distribution(dataset, FOLD, train_loader, val_loader, test_loader)

            best_criterium_value = -float("inf")
            counter = 0
            best_model_state = None
            best_train_metrics = None
            best_val_metrics = None
            optimal_threshold = 0.5
           
            for epoch in range(1, num_epochs + 1):
               
                # Desempacota todas as métricas retornadas
                tr_metrics = train_one_epoch(model, train_loader, unsup_loader, criterion, optimizer, scaler, DATASET_NAME, device, ALPHA)
                tr_loss, tr_loss_sup, tr_loss_coral, tr_acc, tr_sens, tr_spec, tr_f1 = tr_metrics

                val_loss, val_acc, val_sens, val_spec, val_f1, val_auc, best_threshold, val_snd, im_score = validate(
                    model, val_loader, criterion, DATASET_NAME, device, unsup_loader=val_unsup_loader
                )

                test_loss, test_acc, test_sens, test_spec, test_f1, test_auc, best_threshold = validate(
                    model, test_loader, criterion, DATASET_NAME, device, test=True, best_threshold=optimal_threshold
                )
                
                if scheduler is not None:
                    scheduler.step(val_loss)

                logging.info(
                    f"Epoch {epoch:03d} | "
                    f"Tr Loss: {tr_loss:.4f} | Class: {tr_loss_sup:.4f} | Coral: {tr_loss_coral:.4f} | Acc: {tr_acc:.4f} | F1: {tr_f1:.4f} || "
                    f"Val Loss: {val_loss:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f} || SND: {val_snd:.4f} | IM: {im_score:.4f} | F1 TEST: {test_f1:.4f} | AUC TEST: {test_auc:.4f}"
                )

                print_tensorboard(writer, FOLD, DATASET_NAME, tr_loss,tr_loss_coral, tr_f1,  val_f1, val_auc,val_snd,im_score,test_auc, epoch)
                criterium_value = (val_snd + val_auc) / 2

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


            model.load_state_dict(best_model_state)
            test_metrics_raw = validate(model, test_loader, criterion, DATASET_NAME, device, test=True, best_threshold=optimal_threshold)
            
            test_metrics = (test_metrics_raw[0], test_metrics_raw[1], test_metrics_raw[2], test_metrics_raw[3], test_metrics_raw[4], test_metrics_raw[5])
            
            save_results_fold(fold_results, DATASET_NAME, FOLD, best_train_metrics, best_val_metrics, test_metrics)

            if SAVE:
                os.makedirs("checkpoints_CORAL", exist_ok=True)
                path = f"checkpoints_CORAL/{DATASET_NAME}_fold{FOLD}_best.pth"
                torch.save(best_model_state, path)
                logging.info(f"✅ Salvo: {path}")
            
            print_results_fold(FOLD, fold_results)
            
            del model, optimizer, scaler, train_loader, val_loader, unsup_loader
            torch.cuda.empty_cache()
            gc.collect()

    return fold_results