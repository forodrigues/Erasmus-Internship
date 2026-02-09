import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import os
import pandas as pd
import copy
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from monai.transforms import (
    Compose, LoadImaged, ResizeD, ToTensord, RandFlipd, RandRotated,
    RandGaussianNoised, RandAdjustContrastd, RandShiftIntensityd,
    EnsureChannelFirstd, NormalizeIntensityd, Lambdad, CastToTyped, EnsureTyped
)
from monai.data import CacheDataset

# Importa as tuas configurações externas
from config import *
from func import *
from datasets import RuntimeAugDataset

# =============================================================================
# 1. UTILITY FUNCTIONS
# =============================================================================

def cycle(iterable):
    """Permite iterar infinitamente sobre o DataLoader não supervisionado."""
    while True:
        for x in iterable:
            yield x

def calc_entropy(probs):
    """Calcula entropia com estabilidade numérica (1e-8)."""
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

def calculate_snd(features, temperature=0.05):
    """Symmetry-induced Normalization Doublet."""
    if features.shape[0] < 2: return 0.0
    features = F.normalize(features, p=2, dim=1)
    sim_mat = torch.mm(features, features.t())   
    mask = torch.eye(sim_mat.size(0), device=sim_mat.device).bool()
    sim_mat.masked_fill_(mask, -float('inf'))
    p_ij = F.softmax(sim_mat / temperature, dim=1)
    entropy = -torch.sum(p_ij * torch.log(p_ij + 1e-10), dim=1)
    return torch.mean(entropy).item()

def calculate_im_score(logits):
    """Information Maximization Score."""
    probs = torch.sigmoid(logits)
    probs_2class = torch.stack([1.0 - probs, probs], dim=1).squeeze()
    if probs_2class.dim() == 1: probs_2class = probs_2class.unsqueeze(0)
    
    entropy_per_sample = -torch.sum(probs_2class * torch.log(probs_2class + 1e-6), dim=1)
    mean_conditional_entropy = torch.mean(entropy_per_sample)
    mean_probs = torch.mean(probs_2class, dim=0)
    marginal_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-6))
    return (marginal_entropy - mean_conditional_entropy).item()

def print_tensorboard(writer, FOLD, DATASET, lambda_grl, train_loss, class_loss, domain_loss, train_f1, 
                      val_loss, val_sens, val_spec, val_f1, val_auc, val_dom_acc, epoch):
    """Log para Tensorboard incluindo Val_Dom_Acc."""
    writer.add_scalar(f"Fold{FOLD}_{DATASET}/Params/Lambda_GRL", lambda_grl, epoch)
    
    writer.add_scalar(f"Fold{FOLD}_{DATASET}/Train/Total_Loss", train_loss, epoch)
    writer.add_scalar(f"Fold{FOLD}_{DATASET}/Train/Loss_Class", class_loss, epoch)
    writer.add_scalar(f"Fold{FOLD}_{DATASET}/Train/Loss_Domain", domain_loss, epoch)
    writer.add_scalar(f"Fold{FOLD}_{DATASET}/Train/F1", train_f1, epoch)

    writer.add_scalar(f"Fold{FOLD}_{DATASET}/Val/Loss", val_loss, epoch)
    writer.add_scalar(f"Fold{FOLD}_{DATASET}/Val/Dom_Acc", val_dom_acc, epoch) # Nova métrica
    writer.add_scalar(f"Fold{FOLD}_{DATASET}/Val/Sens", val_sens, epoch)
    writer.add_scalar(f"Fold{FOLD}_{DATASET}/Val/Spec", val_spec, epoch)
    writer.add_scalar(f"Fold{FOLD}_{DATASET}/Val/F1", val_f1, epoch)
    writer.add_scalar(f"Fold{FOLD}_{DATASET}/Val/AUC", val_auc, epoch)

# =============================================================================
# 2. TRAINING LOGIC (ONE EPOCH)
# =============================================================================

def train_one_epoch(model, sup_loader, unsup_loader, criterion_cls, optimizer, device, lambda_grl):
    model.train()
    
    total_loss, total_loss_cls, total_loss_dom = 0, 0, 0
    total_samples, total_dom_samples, total_dom_correct = 0, 0, 0
    all_labels, all_probs = [], []
    
    unsup_iter = cycle(unsup_loader)
    domain_criterion = nn.BCEWithLogitsLoss(reduction='none') 
    
    for sup_batch in sup_loader:
        # --- Loading ---
        imgs_s = sup_batch['image'].to(device, non_blocking=True)
        labels_s = sup_batch['label'].to(device, non_blocking=True).float().unsqueeze(1)
        
        unsup_batch = next(unsup_iter)
        imgs_u = unsup_batch['image'].to(device, non_blocking=True)
        
        batch_size_s = imgs_s.size(0)
        batch_size_u = imgs_u.size(0)
        all_imgs = torch.cat((imgs_s, imgs_u), dim=0)
        
        optimizer.zero_grad() 
        
        # --- Forward ---
        all_cls_logits, all_dom_logits = model(all_imgs, lambda_grl) 
        cls_logits_s = all_cls_logits[:batch_size_s]
        
        # --- Classification Loss ---
        alpha = 0.2  # Fator de smoothing
        smoothed_labels = labels_s * (1 - alpha) + 0.5 * alpha

        loss_cls = criterion_cls(cls_logits_s, smoothed_labels)

        #loss_cls = criterion_cls(cls_logits_s, labels_s)

        # --- CDAN Entropy Conditioning ---
        probs_all = torch.sigmoid(all_cls_logits)
        probs_2class = torch.cat((1.0 - probs_all, probs_all), dim=1) 
        entropy = calc_entropy(probs_2class) 
        entropy_weights = (1.0 + torch.exp(-entropy)).detach()
        entropy_weights = entropy_weights / torch.mean(entropy_weights)

        domain_labels_s = torch.zeros((batch_size_s, 1), device=device)
        domain_labels_u = torch.ones((batch_size_u, 1), device=device)
        domain_labels = torch.cat([domain_labels_s, domain_labels_u], dim=0)
        
        loss_dom_batch = domain_criterion(all_dom_logits, domain_labels)
        loss_dom = torch.mean(loss_dom_batch * entropy_weights.unsqueeze(1))

        # --- Optimization ---
        loss = loss_cls + loss_dom
        loss.backward()
        optimizer.step()

        # --- Metrics ---
        probs_s = torch.sigmoid(cls_logits_s)
        total_loss += loss.item() * batch_size_s
        total_loss_cls += loss_cls.item() * batch_size_s
        total_loss_dom += loss_dom.item() * (batch_size_s + batch_size_u)
        total_samples += batch_size_s
        
        dom_preds = (all_dom_logits > 0).float()
        total_dom_correct += (dom_preds == domain_labels).sum().item()
        total_dom_samples += (batch_size_s + batch_size_u)

        all_labels.extend(labels_s.detach().cpu().numpy().astype(int).ravel())
        all_probs.extend(probs_s.detach().cpu().numpy().ravel())

    # Métricas de Treino
    train_acc = accuracy_score(all_labels, (np.array(all_probs) > 0.5).astype(int))
    train_f1 = f1_score(all_labels, (np.array(all_probs) > 0.5).astype(int), average='macro')
    
    cm = confusion_matrix(all_labels, (np.array(all_probs) > 0.5).astype(int), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    train_sens = tp / (tp + fn + 1e-8)
    train_spec = tn / (tn + fp + 1e-8)
    dom_acc = total_dom_correct / total_dom_samples

    return total_loss/total_samples, total_loss_cls/total_samples, total_loss_dom/total_dom_samples, train_acc, train_sens, train_spec, train_f1, dom_acc

# =============================================================================
# 3. VALIDATION LOGIC (Com Domain Acc e Agregação)
# =============================================================================

def validate(model, loader, criterion, DATASET, device, test=False, best_threshold=None, unsup_loader=None):
    model.eval()
    
    total_loss, total_domain_loss = 0, 0
    total_samples_cls, total_samples_dom = 0, 0
    total_dom_correct = 0 # ACUMULADOR DE ACURACIA DE DOMINIO
    
    list_ids, list_labels, list_probs = [], [], []
    list_features_u, list_logits_u = [], []

    unsup_iter = cycle(unsup_loader) if unsup_loader is not None else None 
    criterion_dom = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_s in loader:
            imgs_s = batch_s['image'].to(device, non_blocking=True)
            labels_s = batch_s['label'].to(device, non_blocking=True).float().unsqueeze(1)
            ids_s = batch_s['id']
            bs_s = imgs_s.size(0)
            
            if unsup_iter:
                imgs_u = next(unsup_iter)['image'].to(device, non_blocking=True)
                bs_u = imgs_u.size(0)
                all_imgs = torch.cat((imgs_s, imgs_u), dim=0)
            else:
                all_imgs, bs_u = imgs_s, 0

            # Forward
            if unsup_loader is not None and not test:
                try:
                    cls_p, dom_p, feats = model(all_imgs, lambda_grl=0, return_features=True)
                except:
                    cls_p, dom_p = model(all_imgs, lambda_grl=0)
                    feats = None
            else:
                cls_p, dom_p = model(all_imgs, lambda_grl=0)
                feats = None

            cls_s = cls_p[:bs_s]
            
            # Loss Classificação
            total_loss += criterion(cls_s, labels_s).item() * bs_s
            
            # Loss & Acurácia de Domínio
            dom_labels = torch.cat([torch.zeros(bs_s, 1, device=device), torch.ones(bs_u, 1, device=device)], dim=0) if bs_u > 0 else torch.zeros(bs_s, 1, device=device)
            total_domain_loss += criterion_dom(dom_p, dom_labels).item() * (bs_s + bs_u)
            
            dom_preds = (dom_p > 0).float()
            total_dom_correct += (dom_preds == dom_labels).sum().item() # CALCULO DA ACURACIA

            total_samples_cls += bs_s
            total_samples_dom += (bs_s + bs_u)

            # --- CORRECÇÃO DE IDS ---
            if torch.is_tensor(ids_s):
                list_ids.extend(ids_s.cpu().numpy().tolist())
            else:
                list_ids.extend(list(ids_s))
            
            list_labels.extend(labels_s.cpu().numpy().astype(int).ravel())
            list_probs.extend(torch.sigmoid(cls_s).cpu().numpy().ravel())

            # Features para SND/IM
            if feats is not None and bs_u > 0:
                f_u = feats[bs_s:].mean(dim=(2,3)) if feats.dim() == 4 else feats[bs_s:]
                list_features_u.append(f_u.cpu())
                list_logits_u.append(cls_p[bs_s:].cpu())

    # --- PROCESSAMENTO AGREGADO POR PACIENTE ---
    df_val = pd.DataFrame({'id': list_ids, 'label': list_labels, 'probs': list_probs})
    patient_val = df_val.groupby('id').agg({'label': lambda x: round(x.mean()), 'probs': 'mean'}).reset_index()
    
    y_true = patient_val['label'].to_numpy()
    y_probs = patient_val['probs'].to_numpy()
    p_ids = patient_val['id'].tolist()

    # Cálculo do Threshold
    if best_threshold is None:
        if len(np.unique(y_true)) > 1:
            fpr, tpr, thresholds = roc_curve(y_true, y_probs)
            optimal_idx = np.argmin(np.sqrt((1-tpr)**2 + fpr**2))
            best_threshold = thresholds[optimal_idx]
        else: 
            best_threshold = 0.5

    if test:
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        optimal_idx = np.argmin(np.sqrt((1-tpr)**2 + fpr**2))
        optimal_threshold = thresholds[optimal_idx]

    
    y_pred = (y_probs > best_threshold).astype(int)
    
    # Métricas Finais
    auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)

    avg_loss = total_loss/total_samples_cls if total_samples_cls > 0 else 0
    avg_dom_loss = total_domain_loss/total_samples_dom if total_samples_dom > 0 else 0
    avg_dom_acc = total_dom_correct/total_samples_dom if total_samples_dom > 0 else 0 # MEDIA DA ACURACIA

    if test:
        return avg_loss, avg_dom_loss, avg_dom_acc, acc, sens, spec, f1, auc, optimal_threshold, p_ids, y_true, y_probs
    else:
        snd = calculate_snd(torch.cat(list_features_u).to(device)) if list_features_u else 0
        im = calculate_im_score(torch.cat(list_logits_u).to(device)) if list_logits_u else 0
        return avg_loss, avg_dom_loss, avg_dom_acc, acc, sens, spec, f1, auc, best_threshold, snd, im

# =============================================================================
# 4. MAIN KFOLD LOOP (POOLED)
# =============================================================================

def train_kfold(dataset, DATASET_MANY, ARCHITECTURE, criterion_template, optimizer_template, scheduler_template,
                num_epochs, batch_size, device, patience):

    GAMA_GRL = 10
    MAX_GRL = 1
    
    logging.info(f"Config: CDAN+E | GAMA={GAMA_GRL} | Strategy: Pooled CV + Domain Acc Early Stopping")

    items_list = [item for item in dataset.items]
    df = pd.DataFrame(items_list)
    df["image"] = df["image"].apply(lambda x: os.path.basename(x))
    
    log_dir = f"../Tensorboard/Pooled_CDAN_G{int(GAMA_GRL)}_PatientLevel" 
    writer = SummaryWriter(log_dir=log_dir)

    folds_dict = create_split(df) 
    
    dataset_items = np.array(dataset.items)

    post_spacing_transform = Compose([
        LoadImaged(keys=["image"]), 
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        Lambdad(keys="image", func=aspect_ratio_pres), 
        ResizeD(keys=["image"], spatial_size=(224, 224), mode="bilinear"),
        ToTensord(keys=["image", "label", "subfolder", "id", "image_name", "dataset"]),
        CastToTyped(keys=['image'], dtype=np.float32),
        EnsureTyped(keys=['image'])
    ])

    train_transform = Compose([
        RandFlipd(['image'], spatial_axis=[0], prob=0.3),
        RandRotated(keys=['image'], prob=0.3, range_x=[-0.26, 0.26], mode='bilinear', padding_mode="zeros"),
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

    logging.info("Carregando CacheDataset...")
    full_dataset_cached = CacheDataset(data=dataset_items, transform=post_spacing_transform, cache_rate=1.0)
    
    fold_results = []
    
    for DATASET in df['dataset'].unique():
        
        global_ids, global_y_true, global_y_probs, fold_optimal_thresholds = [], [], [], []
        

        for FOLD in range(K_FOLDS): 
            
                        
            logging.info(f"\n{'='*20} DATASET: {DATASET} | FOLD {FOLD} {'='*20}")
            
            model, optimizer, scheduler, _, params = training_config(ARCHITECTURE)

            classifier_params_ids = list(map(id, model.classifier.parameters()))

            backbone_params = [
                p for p in model.model.parameters() 
                if id(p) not in classifier_params_ids
            ]

            optimizer = torch.optim.AdamW([

                {'params': backbone_params, 'lr': 1e-5, 'weight_decay': 0.05}, 
                {'params': model.classifier.parameters(), 'lr': 1e-4, 'weight_decay': 0.05}, 
                {'params': model.domain_classifier.parameters(), 'lr': 2e-4, 'weight_decay': 1e-3} 
            ])

            train_idx, val_idx, train_idx_final, val_idx_final, test_idx_final = create_set(df, folds_dict, DATASET, FOLD)
            print_overlay(df, train_idx_final, val_idx_final, test_idx_final)


            train_ids_list = df.iloc[train_idx_final]['id'].values.tolist()
            val_ids_list = df.iloc[val_idx_final]['id'].values.tolist()
            test_ids_list = df.iloc[test_idx_final]['id'].values.tolist()

            
            dataset_train = RuntimeAugDataset(full_dataset_cached, train_idx_final, transform=train_transform)
            dataset_unsup = RuntimeAugDataset(full_dataset_cached, train_idx, transform=train_transform)
            dataset_val = RuntimeAugDataset(full_dataset_cached, val_idx_final, transform=val_transform)
            dataset_val_unsup = RuntimeAugDataset(full_dataset_cached, val_idx, transform=val_transform)

            dataset_test = RuntimeAugDataset(full_dataset_cached, test_idx_final, transform=val_transform)

            logging.info(f"---------- Dataset Sizes (Fold {FOLD}) ----------")
            logging.info(f"Train Dataset: {len(dataset_train)} samples")
            logging.info(f"Unsup Train Dataset: {len(dataset_unsup)} samples")
            logging.info(f"Validation Dataset: {len(dataset_val)} samples")
            logging.info(f"Unsup Validation Dataset: {len(dataset_val_unsup)} samples")
            logging.info(f"Test Dataset: {len(dataset_test)} samples")

            dl_kwargs = dict(batch_size=batch_size, num_workers=3, pin_memory=True, persistent_workers=True, prefetch_factor=2)


            train_loader = DataLoader(dataset_train, shuffle=True, drop_last=True, **dl_kwargs)
            unsup_loader = DataLoader(dataset_unsup, shuffle=True, drop_last=True, **dl_kwargs)
            val_loader = DataLoader(dataset_val, shuffle=False, **dl_kwargs)
            val_unsup_loader = DataLoader(dataset_val_unsup, shuffle=False, **dl_kwargs)

            test_loader = DataLoader(dataset_test, shuffle=False, **dl_kwargs)
            print_loader_distribution(DATASET, FOLD, train_loader, val_loader, test_loader)

            train_labels = df.iloc[train_idx_final]['label'].values
            num_neg = (train_labels == 0).sum()
            num_pos = (train_labels == 1).sum()
            pos_weight = torch.tensor([num_neg / max(num_pos, 1e-5)], device=device).float()
            criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            best_criterium_value = -float("inf")
            counter = 0
            best_model_state = None
            best_fold_thresh = 0.5
            best_train_metrics = None
            best_val_metrics = None
            
            for epoch in range(1, num_epochs + 1):
                p = epoch / num_epochs
                lambda_grl = (2 / (1 + np.exp(-GAMA_GRL * p)) - 1) * MAX_GRL
                # 1. Train
                tr_loss, tr_cls_loss, tr_dom_loss, tr_acc, tr_sens, tr_spec, tr_f1, dom_acc = train_one_epoch(
                    model, train_loader, unsup_loader, criterion_cls, optimizer, device, lambda_grl
                )
                
                # 2. Validation (retorna avg_dom_acc agora)
                val_loss, val_dom_loss, val_dom_acc, val_acc, val_sens, val_spec, val_f1, val_auc, epoch_best_thresh, snd, im = validate(
                    model, val_loader, criterion_cls, DATASET, device, test=False, unsup_loader=val_unsup_loader
                )

                _, _, _, _, _, _, test_f1, test_auc, _, _, _ = validate(
                    model, test_loader, criterion_cls, DATASET, device, test=False, best_threshold=None,unsup_loader=unsup_loader
                )
                
                if scheduler: scheduler.step(val_loss)

                logging.info(f"Ep {epoch:02d} | GRL {lambda_grl:.2f} | Tr_Class_Loss { tr_cls_loss:.2f} | Tr_Dom_Loss { tr_dom_loss:.2f} |  Tr_Dom_Acc {dom_acc:.2f} | Tr_F1 {tr_f1:.2f} | Val_Dom_Loss {val_loss:.2f} | Val_Dom_Acc {val_dom_acc:.2f} | Val_Sens {val_sens:.4f} | Val_Spec {val_spec:.4f} |  Val_F1 {val_f1:.4f} | Val_AUC {val_auc:.4f} | Test_AUC {test_auc:.4f}")
                
                print_tensorboard(writer, FOLD, DATASET, lambda_grl, tr_loss, tr_cls_loss, tr_dom_loss, tr_f1, 
                                  val_loss, val_sens, val_spec, val_f1, val_auc, val_dom_acc, epoch)

                # 3. Early Stopping Híbrido (AUC - |Confusão|)
                # Penaliza se o discriminador for muito certeiro (>0.5). Ideal é val_dom_acc = 0.5
                domain_penalty = abs(dom_acc - 0.5)
                criterium = val_auc - 0.5 * domain_penalty
                
                if criterium > best_criterium_value + 1e-4:
                    best_criterium_value = criterium
                    counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_fold_thresh = epoch_best_thresh 
                    
                    best_train_metrics = (tr_loss, tr_acc, tr_sens, tr_spec, tr_f1)
                    best_val_metrics = (val_loss, val_acc, val_sens, val_spec, val_f1, val_auc)
                else:
                    counter += 1
                
                if counter >= patience:
                    logging.info(f"Early stopping no epoch {epoch}")
                    break
                
            
            # --- FIM DO FOLD ---
            model.load_state_dict(best_model_state)
            fold_optimal_thresholds.append(best_fold_thresh)
            
            # Inferência no Teste
            test_loss, test_dom_loss, test_dom_acc, test_acc, test_sens, test_spec, test_f1, test_auc, optimal_threshold, t_ids, t_true, t_probs = validate(
                model, test_loader, criterion_cls, DATASET, device, test=True, best_threshold=best_fold_thresh
            )
            
            logging.info(f"Optimal theshold: {optimal_threshold}")
            logging.info(f"Used theshold: {best_fold_thresh}")

            global_ids.extend(t_ids)
            global_y_true.extend(t_true)
            global_y_probs.extend(t_probs)
            
            test_metrics = (test_loss, test_acc, test_sens, test_spec, test_f1, test_auc)
            save_results_fold(fold_results, DATASET, FOLD, best_train_metrics, best_val_metrics, test_metrics)
            print_results_fold(FOLD, fold_results)
            
            if "SAVE" in globals() and SAVE:
                os.makedirs("checkpoints_pooled", exist_ok=True)
                torch.save(best_model_state, f"checkpoints_pooled/{DATASET}_fold{FOLD}.pth")
                
            del model, optimizer, train_loader, val_loader, test_loader
            torch.cuda.empty_cache()

        # =====================================================================
        # RELATÓRIO POOLED GLOBAL
        # =====================================================================
        logging.info(f"\n{'#'*60}")
        logging.info(f" RELATÓRIO FINAL POOLED: {DATASET}")
        logging.info(f"{'#'*60}")

        g_true = np.array(global_y_true)
        g_probs = np.array(global_y_probs)
        
        if len(np.unique(g_true)) > 1:
            # 1. Calcular o Threshold Ideal Único para todo o dataset pooled
            # Usamos a Distância Euclidiana ao ponto (0,1) que é mais robusta
            fpr_g, tpr_g, thresholds_g = roc_curve(g_true, g_probs)
            dist_g = np.sqrt((1 - tpr_g)**2 + fpr_g**2)
            global_ideal_thresh = thresholds_g[np.argmin(dist_g)]
            
            # 2. Guardar a média aritmética apenas para fins informativos/comparação
            avg_fold_thresh = np.mean(fold_optimal_thresholds)
            
            # 3. Gerar as predições baseadas no NOVO Threshold Ideal Global
            global_auc = roc_auc_score(g_true, g_probs)
            g_pred = (g_probs > avg_fold_thresh).astype(int)
            
            g_acc = accuracy_score(g_true, g_pred)
            g_f1 = f1_score(g_true, g_pred, average='macro')
            
            cm = confusion_matrix(g_true, g_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            g_sens = tp / (tp + fn + 1e-8)
            g_spec = tn / (tn + fp + 1e-8)
            
            logging.info(f"--- RELATÓRIO FINAL POOLED (Ideal Thresh Strategy) ---")
            logging.info(f"Global AUC:             {global_auc:.4f}")
            logging.info(f"Global Ideal Threshold: {global_ideal_thresh:.4f}")
            logging.info(f"Avg Validation Thresh:  {avg_fold_thresh:.4f}")
            logging.info(f"Global F1:  {g_f1:.4f}")

            logging.info(f"Global Accuracy:        {g_acc:.4f}")
            logging.info(f"Global Sens/Spec:       {g_sens:.4f} / {g_spec:.4f}")
            logging.info(f"Global Confusion Matrix:\n{cm}")
            
            df_pooled = pd.DataFrame({
                'patient_id': global_ids,
                'true_label': g_true,
                'prob_score': g_probs
            })
            df_pooled.to_csv(f"pooled_results_{DATASET}.csv", index=False)
        else:
            logging.error("Não foi possível calcular AUC Pooled (apenas 1 classe no target).")

    return fold_results