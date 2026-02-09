import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, cohen_kappa_score, roc_curve, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from collections import Counter
import numpy as np
import logging
from config import *
from func import *
import os
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
import matplotlib.pyplot as plt
from datasets import Dataset2D, RuntimeAugDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import zip_longest
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (Compose, LoadImaged, ToTensord, RandFlipd, RandRotated, 
                              RandZoomd, RandGaussianNoised, RandGaussianSmoothd, 
                              CastToTyped, EnsureTyped, ResizeD, NormalizeIntensityd, 
                              EnsureChannelFirstd)
from monai.data import CacheDataset, set_track_meta
import copy
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


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

def calculate_im_score(logits):
    probs = torch.sigmoid(logits)
    probs_2class = torch.stack([1.0 - probs, probs], dim=1).squeeze()
    
    entropy_per_sample = -torch.sum(probs_2class * torch.log(probs_2class + 1e-6), dim=1)
    mean_conditional_entropy = torch.mean(entropy_per_sample)
    
    mean_probs = torch.mean(probs_2class, dim=0)
    marginal_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-6))
    
    im_score = marginal_entropy - mean_conditional_entropy
    return im_score.item()

def calc_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-30), dim=1)


# -----------------------------------------------------------------------------
# Training Logic (One Epoch) - SEM AUTOCAST
# -----------------------------------------------------------------------------
def train_one_epoch(model, sup_loader, unsup_loader, criterion_cls, optimizer, DATASET, device, lambda_grl, ALPHA=1.0):
    model.train()
    
    total_loss = 0
    total_loss_cls = 0
    total_loss_dom = 0
    total_dom_correct = 0  
    total_dom_samples = 0  
    total_samples = 0
    
    all_ids, all_labels, all_probs = [], [], []
    unsup_iter = cycle(unsup_loader)
    
    # Critério de domínio base
    domain_criterion = nn.BCEWithLogitsLoss(reduction='none') 
    
    for sup_batch in sup_loader:
        
        # --- Data Loading (Source & Target) ---
        imgs_s = sup_batch['image'].to(device, non_blocking=True)
        labels_s = sup_batch['label'].to(device, non_blocking=True).float().unsqueeze(1)
        ids_s = sup_batch['id']
        batch_size_s = imgs_s.size(0)
        
        unsup_batch = next(unsup_iter)
        imgs_u = unsup_batch['image'].to(device, non_blocking=True)
        batch_size_u = imgs_u.size(0)

        all_imgs = torch.cat((imgs_s, imgs_u), dim=0)
        
        optimizer.zero_grad() 

        # --- Forward Pass (Sem Autocast) ---
        all_cls_logits, all_dom_logits = model(all_imgs, lambda_grl) 
        cls_logits_s = all_cls_logits[:batch_size_s]
        
        # --- Task Loss Calculation ---
        loss_cls = criterion_cls(cls_logits_s, labels_s)

        # --- CDAN Entropy Weighting ---
        probs_all_1 = torch.sigmoid(all_cls_logits)
        probs_all_0 = 1.0 - probs_all_1
        probs_2class = torch.cat((probs_all_0, probs_all_1), dim=1) 
        
        entropy = calc_entropy(probs_2class) 
        entropy_weights = 1.0 + torch.exp(-entropy)
        entropy_weights = (entropy_weights / torch.mean(entropy_weights)).detach()

        # --- Domain Loss Calculation ---
        domain_labels_s = torch.zeros(batch_size_s, 1, device=device) # Source = 0
        domain_labels_u = torch.ones(batch_size_u, 1, device=device)  # Target = 1
        all_dom_labels = torch.cat([domain_labels_s, domain_labels_u], dim=0)
        
        loss_dom_unweighted = domain_criterion(all_dom_logits, all_dom_labels)
        loss_dom = torch.mean(loss_dom_unweighted * entropy_weights.unsqueeze(1))

        # --- Optimization (Standard) ---
        loss = loss_cls + loss_dom 
        loss.backward()
        optimizer.step()

        # --- Metrics Accumulation ---
        probs_s = torch.sigmoid(cls_logits_s)
        total_loss += loss.item() * batch_size_s
        total_loss_cls += loss_cls.item() * batch_size_s
        total_loss_dom += loss_dom.item() * (batch_size_s + batch_size_u) 
        total_samples += batch_size_s

        # --- CÁLCULO DA ACURÁCIA DO DOMÍNIO ---
        dom_preds = (all_dom_logits > 0).float() 
        dom_correct = (dom_preds == all_dom_labels).sum().item()
        total_dom_correct += dom_correct
        total_dom_samples += (batch_size_s + batch_size_u)

        all_ids.extend(ids_s.numpy() if not torch.is_tensor(ids_s) else ids_s.cpu().numpy())
        all_labels.extend(labels_s.detach().cpu().numpy().astype(int).ravel())
        all_probs.extend(probs_s.detach().cpu().numpy().ravel())

    # --- Epoch Summary Metrics ---
    df = pd.DataFrame({'patient_id': all_ids, 'label': all_labels, 'probs': all_probs})
    patient_df = df.groupby('patient_id').agg({'label': lambda x: round(x.mean()), 'probs': 'mean'}).reset_index()
    
    y_true = patient_df['label'].to_numpy()
    y_probs = patient_df['probs'].to_numpy()
    
    if len(np.unique(y_true)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        best_threshold_train = thresholds[(tpr - fpr).argmax()]
    else:
        best_threshold_train = 0.5
        
    y_pred = (y_probs > best_threshold_train).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0,0,0,0)
    sens = TP / (TP + FN + 1e-8)
    spec = TN / (TN + FP + 1e-8)

    avg_loss = total_loss / total_samples
    avg_loss_cls = total_loss_cls / total_samples
    avg_loss_dom = total_loss_dom / total_dom_samples 
    avg_dom_acc = total_dom_correct / total_dom_samples

    return avg_loss, avg_loss_cls, avg_loss_dom, acc, sens, spec, f1, avg_dom_acc


# -----------------------------------------------------------------------------
# Validation Logic - SEM AUTOCAST - COM DOMAIN LOSS (LÓGICA TREINO)
# -----------------------------------------------------------------------------

def validate(model, loader, criterion, DATASET, device, test=False, best_threshold=None, unsup_loader=None):
    model.eval()
    
    total_loss = 0          
    total_domain_loss = 0   
    total_samples_cls = 0
    total_samples_dom = 0
    
    list_ids, list_labels, list_probs = [], [], []
    list_features_u, list_logits_u = [], []

    unsup_iter = cycle(unsup_loader) if unsup_loader is not None else None   
    
    # Na validação não aplicamos pesos de entropia na loss, queremos ver a loss real
    criterion_dom = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_s in loader:
            imgs_s = batch_s['image'].to(device, non_blocking=True)
            labels_s = batch_s['label'].to(device, non_blocking=True).float().unsqueeze(1)
            ids_s = batch_s['id']
            bs_s = imgs_s.size(0)
            
            # Lógica de concatenação igual ao treino
            if unsup_iter is not None:
                batch_u = next(unsup_iter)
                imgs_u = batch_u['image'].to(device, non_blocking=True)
                bs_u = imgs_u.size(0)
                all_imgs = torch.cat((imgs_s, imgs_u), dim=0)
            else:
                all_imgs = imgs_s
                bs_u = 0

            # Forward (Sem Autocast)
            if unsup_loader is not None and not test:
                cls_preds_all, dom_preds_all, features_all = model(all_imgs, lambda_grl=0, return_features=True)
            else:
                cls_preds_all, dom_preds_all = model(all_imgs, lambda_grl=0)
                features_all = None

            # Class Loss (Só Source)
            cls_preds_s = cls_preds_all[:bs_s]
            loss_cls = criterion(cls_preds_s, labels_s)
            
            # Domain Loss (Concatenação)
            dom_labels_s = torch.zeros(bs_s, 1, device=device)
            if bs_u > 0:
                dom_labels_u = torch.ones(bs_u, 1, device=device)
                all_dom_labels = torch.cat([dom_labels_s, dom_labels_u], dim=0)
            else:
                all_dom_labels = dom_labels_s
            
            loss_dom = criterion_dom(dom_preds_all, all_dom_labels)

            total_loss += loss_cls.item() * bs_s
            total_domain_loss += loss_dom.item() * (bs_s + bs_u)
            
            total_samples_cls += bs_s
            total_samples_dom += (bs_s + bs_u)

            probs_s = torch.sigmoid(cls_preds_s)
            
            list_ids.extend(ids_s)
            list_labels.append(labels_s.cpu())
            list_probs.append(probs_s.cpu())

            # Guardar features do Target para SND
            if features_all is not None and bs_u > 0:
                feats_u = features_all[bs_s:] 
                logits_u = cls_preds_all[bs_s:]
                
                if feats_u.dim() == 4:
                    feats_u = feats_u.mean(dim=(2, 3))
                
                list_features_u.append(feats_u.cpu())
                list_logits_u.append(logits_u.cpu())

    # --- Processamento Final ---
    all_labels = torch.cat(list_labels).numpy().astype(int).ravel()
    all_probs = torch.cat(list_probs).numpy().ravel()
    
    if len(list_ids) > 0 and isinstance(list_ids[0], torch.Tensor):
        all_ids = torch.stack(list_ids).numpy()
    else:
        all_ids = list_ids

    # Unsupervised Metrics
    snd_score = 0.0
    im_score = 0.0
    if len(list_features_u) > 0:
        all_features_u = torch.cat(list_features_u, dim=0)
        all_logits_u = torch.cat(list_logits_u, dim=0)
        
        snd_score = calculate_snd(all_features_u.to(device)) 
        im_score = calculate_im_score(all_logits_u.to(device))

    # Métricas Supervisionadas
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

    avg_loss = total_loss / total_samples_cls if total_samples_cls > 0 else 0
    avg_dom_loss = total_domain_loss / total_samples_dom if total_samples_dom > 0 else 0

    if test:
        return avg_loss, avg_dom_loss, acc, sens, spec, f1, auc, best_threshold
    else:
        return avg_loss, avg_dom_loss, acc, sens, spec, f1, auc, best_threshold, snd_score, im_score

# -----------------------------------------------------------------------------
# Main K-Fold Training Loop
# -----------------------------------------------------------------------------

def train_kfold(dataset, DATASET_MANY, ARCHITECTURE, criterion, optimizer, scheduler,
                num_epochs, batch_size, device, patience=PATIENCE):

    global K_FOLDS, params, PRE_TRAINED, SAVE
    fold_results = []
    items_list = [item for item in dataset.items]
    df = pd.DataFrame(items_list)
    df["image"] = df["image"].apply(lambda x: os.path.basename(x))
    
    n_splits = 5
    GAMA = 2
    ALPHA = 1
    folds_dict = create_split(df)
    
    log_dir = f"../Tensorboard/CDAN__GAMA{GAMA}" 
    writer = SummaryWriter(log_dir=log_dir)

    dataset_items = np.array(dataset.items)

    # --- Transforms Configuration ---
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
        num_workers=7    
    )
    logging.info(f"Dataset carregado. Tamanho: {len(full_dataset_cached)}")

    # --- Fold Iteration ---
    for DATASET in df['dataset'].unique():
        for FOLD in range(n_splits):

            model, optimizer, scheduler, criterion, params = training_config(ARCHITECTURE)
            optimizer = torch.optim.AdamW([
                {'params': model.model.parameters(), 'lr': 1e-5}, 
                {'params': model.classifier.parameters(), 'lr': 1e-4},             
                {'params': model.domain_classifier.parameters(), 'lr': 1e-3} 
            ], weight_decay=0.01)

            train_idx, val_idx, train_idx_final, val_idx_final, test_idx_final = create_set(df, folds_dict, DATASET, FOLD)
            
            # --- Class Weighting ---
            train_labels = df.iloc[train_idx_final]['label'].values
            num_neg = (train_labels == 0).sum()
            num_pos = (train_labels == 1).sum()
            weight_value = num_neg / max(num_pos, 1e-5) 
            pos_weight = torch.tensor([weight_value], device=device).float()
            criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            print_overlay(df, train_idx_final, val_idx_final, test_idx_final)

            # --- Dataset Splitting ---
            dataset_train = RuntimeAugDataset(full_dataset_cached, train_idx_final, transform=runtime_aug)
            dataset_unsup = RuntimeAugDataset(full_dataset_cached, train_idx, transform=runtime_aug)
            
            logging.info(f"📊 Unsupervised Dataset Size: {len(dataset_unsup)}")
            logging.info(f"📊 Supervised Dataset Size: {len(dataset_train)}")

            dataset_val = Subset(full_dataset_cached, val_idx_final)
            dataset_val_unsup = Subset(full_dataset_cached, val_idx)
            dataset_test = Subset(full_dataset_cached, test_idx_final)
            logging.info(f"📊 Unsupervised Val Dataset Size: {len(dataset_val_unsup)}")

            # --- DataLoaders ---
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

            # --- Epoch Loop ---
            for epoch in range(1, num_epochs + 1):
                
                p = epoch / num_epochs
                lambda_grl = 2 / (1 + np.exp(-GAMA * p)) - 1  

                train_loss, class_loss, domain_loss, train_acc, train_sens, train_spec, train_f1, avg_dom_acc = train_one_epoch(
                    model, train_loader, unsup_loader, criterion_cls, optimizer, DATASET, device, lambda_grl, ALPHA
                )

                # Validação com Domain Loss extraída
                val_loss, val_dom_loss, val_acc, val_sens, val_spec, val_f1, val_auc, best_threshold, snd_score, im_score = validate(
                    model, val_loader, criterion_cls, DATASET, device, unsup_loader=val_unsup_loader
                )

                # Teste com Domain Loss extraída
                test_loss, test_dom_loss, test_acc, test_sens, test_spec, test_f1, test_auc, best_threshold = validate(
                    model, test_loader, criterion_cls, DATASET, device, test=True, best_threshold=optimal_threshold
                )

                if scheduler is not None:
                    scheduler.step(val_loss)

                logging.info(
                    f"Epoch {epoch:03d} | "
                    f"GRL {lambda_grl:.2f} | "
                    f"Tr_Loss: {train_loss:.3f} | Dom_Loss: {domain_loss:.3f} | Dom_Acc: {avg_dom_acc:.3f} | F1: {train_f1:.3f} || "
                    f"Val_Loss: {val_loss:.3f} | Val_Dom: {val_dom_loss:.3f} | F1: {val_f1:.3f} | Test AUC:{val_auc:.4f} | SND: {snd_score:.3f} | IM: {im_score:.4f} | Test_F1 {test_f1:.4f} | Test_Auc {test_auc:.4f} "
                )

                print_tensorboard(writer, FOLD, DATASET, lambda_grl, train_loss, domain_loss, train_f1, val_dom_loss, val_f1, val_auc,snd_score,im_score,test_auc, epoch)
                
                # --- Early Stopping (Com Confusão Máxima + Performance) ---
                # Penaliza se o discriminador for bom demais (loss < 0.69) OU instável (loss > 0.69)
                dist_confusion = abs(val_dom_loss - 0.6931)
                criterium_value = val_auc - (dist_confusion * 0.5)

                if criterium_value > best_criterium_value + 1e-4:
                    best_criterium_value = criterium_value
                    counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_train_metrics = (train_loss, train_acc, train_sens, train_spec, train_f1)
                    best_val_metrics = (val_loss, val_acc, val_sens, val_spec, val_f1, val_auc)
                    optimal_threshold = best_threshold
                else:
                    counter += 1

                if counter >= patience:
                    logging.info(f"Early stopping after {patience} epochs.")
                    break
            
            # --- Final Testing ---
            model.load_state_dict(best_model_state)
            test_loss, test_dom_loss, test_acc, test_sens, test_spec, test_f1, test_auc, best_threshold = validate(
                model, test_loader, criterion_cls, DATASET, device, test=True, best_threshold=optimal_threshold
            )
            test_metrics = (test_loss, test_acc, test_sens, test_spec, test_f1, test_auc)
            
            save_results_fold(fold_results, DATASET, FOLD, best_train_metrics, best_val_metrics, test_metrics)

            if SAVE:
                os.makedirs("checkpoints_CDAN", exist_ok=True)
                model_save_path = f"checkpoints_CDAN/{DATASET}_fold{FOLD}_best.pth"
                torch.save(best_model_state, model_save_path)
                logging.info(f"✅ Melhor modelo do FOLD {FOLD} salvo em: {model_save_path}")

            print_results_fold(FOLD, fold_results)

    return fold_results