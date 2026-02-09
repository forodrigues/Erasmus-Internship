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
from monai.transforms import (Compose, LoadImaged, ToTensord, RandFlipd, RandRotated, 
                              RandZoomd, RandGaussianNoised, RandGaussianSmoothd, 
                              CastToTyped, EnsureTyped, ResizeD, NormalizeIntensityd, 
                              EnsureChannelFirstd)
from monai.data import CacheDataset, set_track_meta
import copy



def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_samples = 0, 0
    all_ids, all_labels, all_probs = [], [], []

    for batch in loader:
        imgs = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True).float().unsqueeze(1)
        ids = batch['id']

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        probs = torch.sigmoid(outputs)
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        all_ids.extend(ids.cpu().numpy())
        all_labels.extend(labels.cpu().numpy().astype(int).ravel())
        all_probs.extend(probs.detach().cpu().numpy().ravel())

    df = pd.DataFrame({'patient_id': all_ids,'label': all_labels,'probs': all_probs})

    patient_df = df.groupby('patient_id').agg({'label': lambda x: round(x.mean()),'probs':'mean'}).reset_index()

    threshold = 0.5
    patient_df['pred'] = (patient_df['probs'] > threshold).astype(int)

    y_true = patient_df['label'].to_numpy()
    y_pred = patient_df['pred'].to_numpy()
    patient_probs = patient_df['probs'].to_numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    sens = TP / (TP + FN + 1e-8)
    spec = TN / (TN + FP + 1e-8)

    return total_loss / total_samples, acc, sens, spec, f1

def validate(model, loader, criterion, device, test=False,best_threshold=0.5):
    model.eval()
    total_loss, total_samples = 0, 0
    all_ids, all_labels, all_probs = [], [], []

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

            all_ids.extend(ids.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().astype(int).ravel())
            all_probs.extend(probs.detach().cpu().numpy().ravel())

    df = pd.DataFrame({'patient_id': all_ids, 'label': all_labels, 'probs': all_probs})

    patient_df = df.groupby('patient_id').agg({'label': lambda x: round(x.mean()),'probs': 'mean'}).reset_index()
    fpr, tpr, thresholds = roc_curve(patient_df['label'], patient_df['probs'])
    best_threshold = thresholds[(tpr - fpr).argmax()]
    patient_df['pred'] = (patient_df['probs'] > best_threshold).astype(int)

    y_true = patient_df['label'].to_numpy()
    y_pred = patient_df['pred'].to_numpy()
    patient_probs = patient_df['probs'].to_numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    sens = TP / (TP + FN + 1e-8)
    spec = TN / (TN + FP + 1e-8)
    auc = roc_auc_score(y_true, patient_probs)

    return total_loss / total_samples, acc, sens, spec, f1, auc, best_threshold



def train_kfold(dataset,DATASET_MANY,ARCHITECTURE, criterion, optimizer, scheduler,
                num_epochs, batch_size, device, patience=PATIENCE):

    global K_FOLDS,params, PRE_TRAINED,SAVE
    fold_results = []
    PERCENTAGE = 1
    #METHOD = "GRADUAL"
    METHOD = "DIFF"
    #METHOD = ""
    #or comment both for fixed LR
    
    logging.info(f"Percentage: {PERCENTAGE}")

    items_list = [item for item in dataset.items]
    df = pd.DataFrame(items_list)
    #df = df.drop("mask_folder", axis=1)
    #df = df.drop("subfolder", axis=1)
    df["image"] = df["image"].apply(lambda x: os.path.basename(x))
    train_dfs = []
    val_dfs = []
    test_dfs = []
    folds_dict = {}
    n_splits = 5

    log_dir = f"../Tensorboard/Partial2_{PERCENTAGE}" 
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

    

    for DATASET in df['dataset'].unique():

        SOURCE_DATASET = [d for d in df['dataset'].unique() if d != DATASET][0]

        for FOLD in range(n_splits):

            if FOLD <=3:
                continue
            model, optimizer, scheduler, criterion, params = training_config(ARCHITECTURE)

            checkpoint_path = f"checkpointsalpha1/{SOURCE_DATASET}_fold{FOLD}_best.pth"
            logging.info(checkpoint_path)

            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict, strict=False)


            train_idx_final,val_idx_final,test_idx_final = create_set(df,PERCENTAGE,folds_dict,DATASET,FOLD)


            print_overlay (df,train_idx_final,val_idx_final,test_idx_final)

                
            dataset_train = RuntimeAugDataset(full_dataset_cached, train_idx_final, transform=runtime_aug)
            dataset_val = Subset(full_dataset_cached, val_idx_final)
            dataset_test = Subset(full_dataset_cached, test_idx_final)

            dl_kwargs = dict(
                batch_size=batch_size, 
                num_workers=3,          
                pin_memory=True,        
                persistent_workers=True,
                prefetch_factor=2       
            )

            train_loader = DataLoader(dataset_train, shuffle=True, **dl_kwargs)
            val_loader = DataLoader(dataset_val, shuffle=False, **dl_kwargs)
            test_loader = DataLoader(dataset_test, shuffle=False, **dl_kwargs)

            print_loader_distribution(dataset,FOLD,train_loader,val_loader,test_loader)

            best_criterium_value = -float("inf")
            counter = 0
            best_model_state = None
            best_train_metrics = None
            best_val_metrics = None

            train_labels = df.iloc[train_idx_final]['label'].values
            num_neg = (train_labels == 0).sum()
            num_pos = (train_labels == 1).sum()
            weight_value = num_neg / max(num_pos, 1e-5) 
            pos_weight = torch.tensor([weight_value], device=device).float()
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


            if METHOD == "DIFF":
                backbone_params = model.model.get_classifier().parameters() # Parâmetros da head
                all_params = set(model.parameters())
                head_params = set(model.model.get_classifier().parameters())
                backbone_params = list(all_params - head_params)

                optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': 1e-5},         
                    {'params': list(head_params), 'lr': 1e-4}        
                ], weight_decay=1e-5)

                
            if METHOD == "GRADUAL":
                EPOCHS_PER_UNFREEZE = 5

                for param in model.parameters():
                    param.requires_grad = False

                feature_blocks = model.model.stages 
                next_unfreeze_idx = len(feature_blocks) - 1
                model.model.head.requires_grad_(True)

                
                def update_optimizer(model, feature_blocks):
                    params_to_train = [
                        {'params': model.model.head.parameters(), 'lr': 5e-5},
                        {'params': feature_blocks[3].parameters(), 'lr': 1e-5}, # Estágio final (High-level features)
                        {'params': feature_blocks[2].parameters(), 'lr': 1e-5},
                        {'params': feature_blocks[1].parameters(), 'lr': 5e-6},
                        {'params': feature_blocks[0].parameters(), 'lr': 1e-6}, # Estágio inicial (Low-level features)
                    ]
                    active_params = [group for group in params_to_train if any(p.requires_grad for p in group['params'])]
                    return torch.optim.AdamW(active_params, weight_decay=1e-5)
                
                optimizer = update_optimizer(model, feature_blocks)

                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logging.info(f"📊 Parâmetros treináveis atuais: {trainable_params:,}")

            for epoch in range(1, num_epochs + 1):

                if METHOD == "GRADUAL":
                    if epoch % EPOCHS_PER_UNFREEZE == 0 and next_unfreeze_idx >= 0:
                        feature_blocks[next_unfreeze_idx].requires_grad_(True)
                        next_unfreeze_idx -= 1
                        logging.info(f"Descongelando bloco {next_unfreeze_idx+1} do feature extractor (reverso)")
                        optimizer = update_optimizer(model, feature_blocks)
                        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        logging.info(f"📊 Parâmetros treináveis atuais: {trainable_params:,}")


                train_loss, train_acc, train_sens, train_spec, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)

                val_loss, val_acc, val_sens, val_spec, val_f1, val_auc, best_threshold  = validate(model, val_loader, criterion, device)

                if scheduler is not None:
                    scheduler.step(val_loss)

                logging.info(
                    f"Epoch {epoch:03d} | "
                    f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Sens: {train_sens:.4f} | Spec: {train_spec:.4f} | F1: {train_f1:.4f} || "
                    f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Sens: {val_sens:.4f} | Spec: {val_spec:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}"
                )

                print_tensorboard(writer,FOLD,DATASET, train_loss,train_f1, val_loss, val_sens,val_spec,val_f1,val_auc,epoch)

                criterium_value = val_auc

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


            model.load_state_dict(best_model_state)
            test_loss, test_acc, test_sens, test_spec, test_f1, test_auc, best_threshold = validate(model, test_loader, criterion, device,optimal_threshold,True)
            test_metrics = (test_loss, test_acc, test_sens, test_spec, test_f1, test_auc)
            save_results_fold (fold_results,DATASET,FOLD,best_train_metrics,best_val_metrics,test_metrics)


            if SAVE:
                os.makedirs("checkpointsalpha1dn", exist_ok=True)
                model_save_path = f"checkpointsalpha1dn/{DATASET}_fold{FOLD}_best.pth"
                torch.save(best_model_state, model_save_path)
                logging.info(f"✅ Melhor modelo do FOLD {FOLD} salvo em: {model_save_path}")

            
            print_results_fold(FOLD,fold_results)

    return fold_results
