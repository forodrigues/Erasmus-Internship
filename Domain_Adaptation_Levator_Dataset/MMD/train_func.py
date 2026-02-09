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
import torch.nn.functional as F
import gc
# Desliga rastreio de metadados para economizar RAM e CPU
set_track_meta(False)

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
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    
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
    Calcula a Loss MMD de forma robusta a batch sizes diferentes.
    """
    source_n = int(source.size()[0])
    target_n = int(target.size()[0])
    
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, 
                              kernel_num=kernel_num, 
                              fix_sigma=fix_sigma)
    
    XX = kernels[:source_n, :source_n]
    YY = kernels[source_n:, source_n:]
    XY = kernels[:source_n, source_n:]
    
    loss = torch.mean(XX) + torch.mean(YY) - 2 * torch.mean(XY)
    
    return loss

# ==============================================================================
# FUNÇÃO DE TREINO COM AMP (Mixed Precision)
# ==============================================================================

def train_one_epoch(model, sup_loader, unsup_loader, criterion, optimizer, scaler, DATASET, device, alpha):

    model.train()
    total_loss, total_loss_sup, total_loss_mmd = 0, 0, 0
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

        with torch.cuda.amp.autocast():
            # Retorna logits e features (flattened)
            outputs_sup, features_sup = model(imgs_sup, return_features=True)
            loss_sup = criterion(outputs_sup, labels)

            loss_mmd = torch.tensor(0.0, device=device)
            loss = loss_sup 
            
            if alpha > 0:
                try:
                    unsup_batch = next(unsup_iter)
                except StopIteration:
                    unsup_iter = iter(unsup_loader)
                    unsup_batch = next(unsup_iter)
                
                imgs_unsup = unsup_batch['image'].to(device, non_blocking=True)
                
                # Para mmd, precisamos que o batch size do target > 1
                if imgs_unsup.size(0) > 1:
                    _, features_unsup = model(imgs_unsup, return_features=True)
                    
                    # Certificar que as features são 2D (Batch, Dim)
                    if len(features_sup.shape) > 2:
                        features_sup = torch.flatten(features_sup, 1)
                    if len(features_unsup.shape) > 2:
                        features_unsup = torch.flatten(features_unsup, 1)

                    #features_sup = F.normalize(features_sup, p=2, dim=1)
                    #features_unsup = F.normalize(features_unsup, p=2, dim=1)
                    loss_mmd = mmd_loss(features_sup, features_unsup)
                    loss = loss_sup + alpha * loss_mmd 
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        probs = torch.sigmoid(outputs_sup)
        bs_sup = labels.size(0)
        
        total_loss += loss.item() * bs_sup
        total_loss_sup += loss_sup.item() * bs_sup
        total_loss_mmd += loss_mmd.item() * bs_sup 
        total_samples += bs_sup

        if isinstance(ids, list):
            all_ids.extend(ids)
        elif torch.is_tensor(ids):
            all_ids.extend(ids.detach().cpu().numpy().tolist())
        else:
            all_ids.extend(np.array(ids).tolist())


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
    avg_loss_mmd = total_loss_mmd / total_samples if total_samples > 0 else 0.0

    return avg_loss, avg_loss_sup, avg_loss_mmd, acc, sens, spec, f1

def validate(model, loader, criterion, DATASET, device, test=False, best_threshold=None, unsup_loader=None):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_ids, all_labels, all_probs = [], [], []
    
    # Listas para acumular features e calcular métricas de domínio
    source_features_list = []
    target_features_list = []
    target_logits_list = []

    with torch.no_grad():
        # --- Parte 1: Domínio Fonte (Validação com Labels - Adela) ---
        for batch in loader:
            imgs = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True).float().unsqueeze(1)
            ids = batch['id']
            
            # Extraímos logits e features
            logits, features = model(imgs, return_features=True)
            probs = torch.sigmoid(logits)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            # Guardar para métricas de alinhamento e agregação
            source_features_list.append(features.cpu())
            
            if isinstance(ids, list):
                all_ids.extend(ids)
            elif torch.is_tensor(ids):
                all_ids.extend(ids.detach().cpu().numpy().tolist())
            else:
                all_ids.extend(np.array(ids).tolist())

            all_labels.extend(labels.cpu().numpy().astype(int).ravel())
            all_probs.extend(probs.cpu().numpy().ravel())

    # --- Parte 2: Domínio Alvo (Validação não supervisionada - Sydney) ---
    snd_score = 0.0
    im_score = 0.0
    val_mmd_alignment = 0.0
    
    if unsup_loader is not None:
        with torch.no_grad():
            for batch in unsup_loader:
                imgs = batch['image'].to(device, non_blocking=True)
                logits, features = model(imgs, return_features=True)
                
                target_features_list.append(features.cpu())
                target_logits_list.append(logits.cpu())
        
        if len(target_features_list) > 0 and len(source_features_list) > 0:
            all_features_source = torch.cat(source_features_list, dim=0)
            all_features_target = torch.cat(target_features_list, dim=0)
            all_logits_target = torch.cat(target_logits_list, dim=0)

            # 1. SND (Densidade de Vizinhança no Alvo)
            snd_score = calculate_snd(all_features_target)
            
            # 2. IM Score (Certeza e Diversidade no Alvo)
            im_score = calculate_im_score(all_logits_target)
            
            features_sup = F.normalize(all_features_source, p=2, dim=1)
            features_unsup = F.normalize(all_features_target, p=2, dim=1)
            val_mmd_alignment = mmd_loss(all_features_source, all_features_target).item()

    # --- Agregação ao nível do Paciente ---
    df = pd.DataFrame({'patient_id': all_ids, 'label': all_labels, 'probs': all_probs})
    patient_df = df.groupby('patient_id').agg({
        'label': lambda x: round(x.mean()), 
        'probs': 'mean'
    }).reset_index()

    y_true = patient_df['label'].to_numpy()
    y_probs = patient_df['probs'].to_numpy()

    # Definição do Threshold Ótimo
    if best_threshold is None:
        if len(np.unique(y_true)) > 1:
            fpr, tpr, thresholds = roc_curve(y_true, y_probs)
            best_threshold = thresholds[(tpr - fpr).argmax()]
        else:
            best_threshold = 0.5
    
    y_pred = (y_probs > best_threshold).astype(int)
    
    # Métricas Clínicas
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = 0.5

    avg_loss = total_loss / total_samples if total_samples > 0 else 0

    if test:
        # No teste individual do fold, não precisamos das métricas de DA
        return avg_loss, acc, sens, spec, f1, auc, best_threshold
    else:
        # Na validação de época, retornamos tudo para monitorização
        return avg_loss, acc, sens, spec, f1, auc, best_threshold, snd_score, im_score, val_mmd_alignment
# ==============================================================================
# FUNÇÃO PRINCIPAL (OTIMIZADA PARA mmd)
# ==============================================================================
def train_kfold(dataset, DATASET_MANY, ARCHITECTURE, criterion, optimizer_cls, scheduler_cls,
                num_epochs, batch_size, device, patience=PATIENCE):

    global K_FOLDS, params, PRE_TRAINED, SAVE
    fold_results = []

    items_list = [item for item in dataset.items]
    df = pd.DataFrame(items_list)
    df["image"] = df["image"].apply(lambda x: os.path.basename(x))
    
    ALPHA = 0
    logging.info(f"Alpha: {ALPHA} | Strategy: Pooled Cross-Validation (Domain Adaptation)")
    
    log_dir = f"../Tensorboard/mmd_unsup2_{ALPHA}" 
    writer = SummaryWriter(log_dir=log_dir)

    folds_dict = create_split(df)
    dataset_items = np.array(dataset.items)
    
    # --- MONAI CacheDataset Setup ---
    logging.info("Iniciando carregamento do CacheDataset (RAM)...")

    
    post_spacing_transform = Compose([
        LoadImaged(keys=["image"]), 
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        Lambdad(keys="image", func=aspect_ratio_pres), 
        ResizeD(keys=["image"], spatial_size=(224, 224), mode="bilinear"),
        ToTensord(keys=["image", "label", "subfolder", "id", "image_name","dataset"]),
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

    full_dataset_cached = CacheDataset(
        data=dataset_items, 
        transform=post_spacing_transform, 
        cache_rate=1  
    )
    logging.info(f"Dataset carregado. Tamanho: {len(full_dataset_cached)}")
    
    for DATASET_NAME in df['dataset'].unique():
        
        # --- VARIÁVEIS PARA ACUMULAÇÃO POOLED ---
        global_ids = []
        global_y_true = []
        global_y_probs = []
        fold_optimal_thresholds = [] # Para calcular o Average Threshold no final

        for FOLD in range(K_FOLDS):
            logging.info(f"\n{'='*40} FOLD {FOLD} {'='*40}")
            
            model, optimizer, scheduler, criterion, params = training_config(ARCHITECTURE)
            model = model.to(device)
            scaler = torch.cuda.amp.GradScaler()

            # Lógica de Split Original
            train_idx, val_idx, train_idx_final, val_idx_final, test_idx_final = create_set(df, folds_dict, DATASET_NAME, FOLD)
            print_overlay(df, train_idx_final, val_idx_final, test_idx_final)

            # Datasets e Loaders (Mantendo a tua estrutura de runtime_aug)
            dataset_train = RuntimeAugDataset(full_dataset_cached, train_idx_final, transform=train_transform)
            dataset_unsup = RuntimeAugDataset(full_dataset_cached, train_idx, transform=train_transform)
            dataset_val = RuntimeAugDataset(full_dataset_cached, val_idx_final, transform=val_transform)
            dataset_val_unsup = RuntimeAugDataset(full_dataset_cached, val_idx, transform=val_transform)
            dataset_test =RuntimeAugDataset(full_dataset_cached, test_idx_final, transform=val_transform)

            dl_kwargs = dict(batch_size=batch_size, num_workers=3, pin_memory=True, persistent_workers=True, prefetch_factor=2)
            train_loader = DataLoader(dataset_train, shuffle=True, **dl_kwargs)
            unsup_loader = DataLoader(dataset_unsup, shuffle=True, **dl_kwargs)
            val_loader = DataLoader(dataset_val, shuffle=False, **dl_kwargs)
            val_unsup_loader = DataLoader(dataset_val_unsup, shuffle=False, **dl_kwargs)
            test_loader = DataLoader(dataset_test, shuffle=False, **dl_kwargs)

            print_loader_distribution(DATASET_NAME, FOLD, train_loader, val_loader, test_loader)

            best_criterium_value = -float("inf")
            counter = 0
            best_model_state = None
            current_fold_optimal_thresh = 0.5 

            for epoch in range(1, num_epochs + 1):
                # Treino com mmd
                tr_metrics = train_one_epoch(model, train_loader, unsup_loader, criterion, optimizer, scaler, DATASET_NAME, device, ALPHA)
                tr_loss, tr_loss_sup, tr_loss_mmd, tr_acc, tr_sens, tr_spec, tr_f1 = tr_metrics

                # Validação (SND + AUC)
                v_metrics = validate(model, val_loader, criterion, DATASET_NAME, device, unsup_loader=val_unsup_loader)
                val_loss, val_acc, val_sens, val_spec, val_f1, val_auc, val_thresh, val_snd, im_score,val_mmd = v_metrics

                test_loss, test_acc, test_sens, test_spec, test_f1, test_auc, best_threshold = validate(model, test_loader, criterion, DATASET_NAME, device, test=True, best_threshold=0.5)
                
                if scheduler is not None: scheduler.step(val_loss)

                logging.info(f"Epoch {epoch:03d} | Tr_Loss: {tr_loss:.4f} | Tr_Loss_mmd: {tr_loss_mmd:.4f} | Val_SND: {val_snd:.4f} | Val_IM: {im_score:.4f} | Val_mmd: {val_mmd:.4f} | Val_AUC: {val_auc:.4f}| Test_AUC: {test_auc:.4f}")

                # Critério de Seleção Híbrido (Source AUC + Target SND)
                criterium_value = val_auc - 0.1 * val_mmd

                if criterium_value > best_criterium_value + 1e-4 and epoch >2:
                    best_criterium_value = criterium_value
                    counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                    current_fold_optimal_thresh = val_thresh # Guarda o thresh da melhor época
                    
                    best_train_metrics = (tr_loss, tr_acc, tr_sens, tr_spec, tr_f1)
                    best_val_metrics = (val_loss, val_acc, val_sens, val_spec, val_f1, val_auc)
                else:
                    counter += 1

                if counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break

            # --- FIM DO FOLD: Avaliação de Teste e Acumulação ---
            model.load_state_dict(best_model_state)
            fold_optimal_thresholds.append(current_fold_optimal_thresh)
            
            # 1. Teste Individual do Fold (Honest Thresholding)
            test_metrics_raw = validate(model, test_loader, criterion, DATASET_NAME, device, test=True, best_threshold=current_fold_optimal_thresh)
            test_metrics = (test_metrics_raw[0], test_metrics_raw[1], test_metrics_raw[2], test_metrics_raw[3], test_metrics_raw[4], test_metrics_raw[5])
            
            save_results_fold(fold_results, DATASET_NAME, FOLD, best_train_metrics, best_val_metrics, test_metrics)

            # 2. Acumulação para o Relatório Pooled Final
            p_ids, p_true, p_probs = get_patient_results(model, test_loader, device)
            global_ids.extend(p_ids)
            global_y_true.extend(p_true)
            global_y_probs.extend(p_probs)

            if SAVE:
                os.makedirs("checkpoints", exist_ok=True)
                path = f"checkpoints/{DATASET_NAME}_fold{FOLD}_best.pth"
                torch.save(best_model_state, path)

            print_results_fold(FOLD, fold_results)
            
            del model, optimizer, scaler, train_loader, val_loader, unsup_loader
            torch.cuda.empty_cache()
            gc.collect()

        # === RELATÓRIO POOLED (GLOBAL) PARA O DATASET ===
        logging.info(f"\n{'*'*60}\n RELATÓRIO POOLED FINAL - DATASET: {DATASET_NAME}\n{'*'*60}")
        
        y_true_pool = np.array(global_y_true)
        y_probs_pool = np.array(global_y_probs)
        
        if len(np.unique(y_true_pool)) > 1:
            global_auc = roc_auc_score(y_true_pool, y_probs_pool)
            
            # Threshold Honesto: Média dos limiares ótimos de cada fold
            avg_val_thresh = np.mean(fold_optimal_thresholds)
            y_pred_pool = (y_probs_pool > avg_val_thresh).astype(int)
            
            global_acc = accuracy_score(y_true_pool, y_pred_pool)
            global_f1 = f1_score(y_true_pool, y_pred_pool, average='macro')
            cm = confusion_matrix(y_true_pool, y_pred_pool, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            
            logging.info(f"Global AUC (Pooled): {global_auc:.4f}")
            logging.info(f"Global F1 (Avg Thresh): {global_f1:.4f}")
            logging.info(f"Global Accuracy: {global_acc:.4f}")
            logging.info(f"Global Sens/Spec: {tp/(tp+fn+1e-8):.4f} / {tn/(tn+fp+1e-8):.4f}")
            logging.info(f"Average Val Threshold used: {avg_val_thresh:.4f}")
            logging.info(f"Global Confusion Matrix:\n{cm}")
            
            # Salvar resultados detalhados
            df_pooled = pd.DataFrame({'id': global_ids, 'true': y_true_pool, 'prob': y_probs_pool, 'pred': y_pred_pool})
            df_pooled.to_csv(f"pooled_results_DA_{DATASET_NAME}.csv", index=False)
        else:
            logging.error(f"Erro Pooled: Apenas uma classe presente nos dados acumulados de {DATASET_NAME}")

    return fold_results